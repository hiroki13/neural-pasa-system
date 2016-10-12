import numpy as np
import theano

from ..ling.vocab import Vocab
from ..ling.sample import Sample
from io_utils import CorpusLoader, say, dump_data, load_data
from stats import corpus_statistics, sample_statistics


class Preprocessor(object):

    def __init__(self, vocab_word, vocab_label, batch_size, window, mp):
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.batch_size = batch_size
        self.window = window
        self.mp = mp

    def create_samples(self, corpus, test=False):
        """
        :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None

        # 1D: n_docs * n_sents, 2D: n_words; elem=Word
        corpus = [sent for doc in corpus for sent in doc]

        if test is False:
            np.random.shuffle(corpus)
            corpus.sort(key=lambda s: len(s))

        samples = []
        for sent in corpus:
            sample = Sample(sent=sent, window=self.window)
            sample.set_params(self.vocab_word, self.vocab_label)
            samples.append(sample)

        return samples

    def get_shared_samples(self, samples):
        """
        :param samples: 1D: n_sents; Sample
        """

        def shared(_sample):
            return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

        batch_size = self.batch_size
        mp = self.mp

        theano_x = []
        theano_y = []
        sent_length = []
        num_prds = []
        batch_index = []

        samples = [sample for sample in samples if sample.n_prds > 0]
        prev_n_prds = samples[0].n_prds
        prev_n_words = samples[0].n_words
        prev_index = 0
        index = 0

        for sample in samples:
            n_prds = sample.n_prds
            n_words = sample.n_words

            #################################
            # Check the boundary of batches #
            #################################
            if prev_n_words != n_words or index - prev_index > batch_size or (n_prds != prev_n_prds and mp):
                batch_index.append((prev_index, index))
                num_prds.append(prev_n_prds)
                sent_length.append(prev_n_words)
                prev_index = index
                prev_n_prds = n_prds
                prev_n_words = n_words

            theano_x.extend(sample.x)
            theano_y.extend(sample.y)
            index += len(sample.x)

        if index > prev_index:
            batch_index.append((prev_index, index))
            num_prds.append(prev_n_prds)
            sent_length.append(prev_n_words)

        assert len(batch_index) == len(sent_length) == len(num_prds)
        return [shared(theano_x), shared(theano_y), shared(sent_length), shared(num_prds)], batch_index


class Experimenter(object):

    def __init__(self, argv):
        self.argv = argv
        self.corpus_loader = None
        self.preprocessor = None

    def select_corpus_loader(self):
        self.corpus_loader = CorpusLoader(min_unit='word', data_size=self.argv.data_size)

    def select_preprocessor(self, vocab_word, vocab_label):
        self.preprocessor = Preprocessor(vocab_word=vocab_word,
                                         vocab_label=vocab_label,
                                         batch_size=self.argv.batch_size,
                                         window=self.argv.window,
                                         mp=False)

    def load_corpus_set(self):
        # corpus: 1D: n_sents, 2D: n_words, 3D: Word()
        train_corpus = self.corpus_loader.load_corpus(path=self.argv.train_data)
        dev_corpus = self.corpus_loader.load_corpus(path=self.argv.dev_data)
        test_corpus = self.corpus_loader.load_corpus(path=self.argv.test_data)
        return train_corpus, dev_corpus, test_corpus

    def create_sample_set(self, corpus_set):
        # samples: 1D: n_sents; Sample
        train_corpus, dev_corpus, test_corpus = corpus_set
        train_samples = self.preprocessor.create_samples(train_corpus)
        dev_samples = self.preprocessor.create_samples(dev_corpus, test=True)
        test_samples = self.preprocessor.create_samples(test_corpus, test=True)
        return train_samples, dev_samples, test_samples

    def create_shared_samples(self, samples):
        return self.preprocessor.get_shared_samples(samples)

    def create_vocab_label(self):
        vocab_label = Vocab()
        vocab_label.set_pas_labels()
        if self.argv.save:
            dump_data(vocab_label, 'vocab_label')
        say('\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i)))
        return vocab_label

    def create_vocab_word(self, corpus):
        vocab_word = Vocab()
        vocab_word.set_init_word()
        vocab_word.add_vocab_from_corpus(corpus=corpus, vocab_cut_off=self.argv.vocab_cut_off)
        if self.argv.save:
            dump_data(vocab_word, 'vocab_word.cut-%d' % self.argv.vocab_cut_off)
        say('\nVocab: %d\tType: word\n' % vocab_word.size())
        return vocab_word

    def load_config(self):
        config = load_data(self.argv.load_config)
        say('\nLoaded config\n')
        say(str(config))
        return config

    def load_labels(self):
        vocab_label = load_data(self.argv.label)
        say('\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i)))
        return vocab_label

    def load_words(self):
        vocab_word = load_data(self.argv.word)
        say('\nVocab: %d\tType: word\n' % vocab_word.size())
        return vocab_word

    @staticmethod
    def show_corpus_stats(corpora):
        for corpus in corpora:
            corpus_statistics(corpus)

    @staticmethod
    def show_sample_stats(sample_set, vocab_label):
        for samples in sample_set:
            sample_statistics(samples, vocab_label)
