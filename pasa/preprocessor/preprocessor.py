import numpy as np
import theano

from sample_factory import BasicSampleFactory, RankingSampleFactory, RerankingSampleFactory, GridSampleFactory
from ..ling.vocab import Vocab, UNK, PAD
from ..utils.io_utils import CorpusLoader, say, load_init_emb
from ..utils.stats import corpus_statistics, sample_statistics


class Preprocessor(object):

    def __init__(self, argv, config=None):
        self.argv = argv
        self.window = self.set_window_size(argv, config)
        self.corpus_loader = None
        self.sample_factory = None

    @staticmethod
    def set_window_size(argv, config):
        if config is not None:
            return config.window
        return argv.window

    def set_corpus_loader(self):
        self.corpus_loader = CorpusLoader(min_unit='word', data_size=self.argv.data_size)

    def set_sample_factory(self, vocab_word, vocab_label):
        self.sample_factory = BasicSampleFactory(vocab_word=vocab_word,
                                                 vocab_label=vocab_label,
                                                 batch_size=self.argv.batch_size,
                                                 window_size=self.window)

    def load_corpus_set(self):
        # corpus: 1D: n_sents, 2D: n_words, 3D: Word()
        train_corpus = self.corpus_loader.load_corpus(path=self.argv.train_data)
        dev_corpus = self.corpus_loader.load_corpus(path=self.argv.dev_data)
        test_corpus = self.corpus_loader.load_corpus(path=self.argv.test_data)
        return train_corpus, dev_corpus, test_corpus

    def create_sample_set(self, corpus_set):
        # samples: 1D: n_sents; Sample
        train_corpus, dev_corpus, test_corpus = corpus_set
        train_samples = self.create_samples(self._flatten(train_corpus))
        dev_samples = self.create_samples(self._flatten(dev_corpus))
        test_samples = self.create_samples(self._flatten(test_corpus))
        return train_samples, dev_samples, test_samples

    @staticmethod
    def _flatten(corpus):
        if corpus:
            return [sample for doc in corpus for sample in doc]
        return None

    def create_samples(self, corpus):
        return self.sample_factory.create_samples(corpus)

    def create_batched_samples(self, samples, n_inputs):
        return self.sample_factory.create_batched_samples(samples, n_inputs)

    @staticmethod
    def create_vocab_label():
        vocab_label = Vocab()
        vocab_label.set_pas_labels()
        say('\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i)))
        return vocab_label

    def create_vocab_word(self, corpus):
        vocab_word = Vocab()
        vocab_word.set_init_word()
        vocab_word.add_vocab_from_corpus(corpus=corpus, vocab_cut_off=self.argv.vocab_cut_off)
        vocab_word.add_word(UNK)
        say('\nVocab: %d\tType: word\n' % vocab_word.size())
        return vocab_word

    def load_init_emb(self):
        vocab_word, emb = load_init_emb(self.argv.init_emb, self.argv.dim_emb)
        say('\n\tWord Embedding Size: %d\n' % vocab_word.size())
        return vocab_word, emb

    @staticmethod
    def show_corpus_stats(corpora):
        for corpus in corpora:
            corpus_statistics(corpus)

    @staticmethod
    def show_sample_stats(sample_set, vocab_label):
        for samples in sample_set:
            sample_statistics(samples, vocab_label)

    def create_trainable_emb(self, train_corpus, vocab_word, emb):
        untrainable_emb = None
        if emb is None:
            vocab_word = self.create_vocab_word(train_corpus)
        elif emb is not None and self.argv.fix == 0:
            vocab_word, emb, untrainable_emb = self._divide_emb(train_corpus, vocab_word, emb)
        return vocab_word, emb, untrainable_emb

    def _divide_emb(self, corpus, vocab_emb_word, emb):
        say('\nDivide the embeddings into the trainable/untrainable embeddings\n')

        vocab_trainable_word = self._get_common_vocab(corpus, vocab_emb_word)
        vocab_trainable_word.add_word(UNK)
        vocab_untrainable_word = Vocab()
        trainable_emb = [[] for i in xrange(vocab_trainable_word.size())]
        untrainable_emb = []

        for w, w_id in vocab_emb_word.w2i.items():
            # TODO: change 'PAD' to '<PAD>'
            if w == PAD:
                continue
            if vocab_trainable_word.has_key(w):
                trainable_emb[vocab_trainable_word.get_id(w)] = emb[w_id]
            else:
                untrainable_emb.append(emb[w_id])
                vocab_untrainable_word.add_word(w)

        vocab_word = self._unite_vocab_word(vocab_trainable_word, vocab_untrainable_word)
        trainable_emb = np.asarray(trainable_emb, theano.config.floatX)
        untrainable_emb = np.asarray(untrainable_emb, theano.config.floatX)

        say('\tTrainable emb: %d  Untrainable emb: %d\n' % (trainable_emb.shape[0], untrainable_emb.shape[0]))
        say('Vocab size: %d  Trainable: %d  Untrainable: %d' % (vocab_word.size(),
                                                                vocab_trainable_word.size(),
                                                                vocab_untrainable_word.size()))
        assert vocab_word.size() == (trainable_emb.shape[0] + untrainable_emb.shape[0] + 1)

        return vocab_word, trainable_emb, untrainable_emb

    @staticmethod
    def _get_common_vocab(corpus, vocab_emb_word):
        vocab_word = Vocab()
        for doc in corpus:
            for sent in doc:
                for w in sent:
                    if vocab_emb_word.has_key(w.form):
                        vocab_word.add_word(w.form)
        return vocab_word

    @staticmethod
    def _unite_vocab_word(vocab_word_1, vocab_word_2):
        vocab_word = Vocab()
        vocab_word.set_init_word()
        for w in vocab_word_1.i2w:
            vocab_word.add_word(w)
        for w in vocab_word_2.i2w:
            vocab_word.add_word(w)
        return vocab_word


class RankingPreprocessor(Preprocessor):

    def __init__(self, argv):
        super(RankingPreprocessor, self).__init__(argv)

    def set_sample_factory(self, vocab_word, vocab_label):
        self.sample_factory = RankingSampleFactory(vocab_word=vocab_word,
                                                   vocab_label=vocab_label,
                                                   batch_size=self.argv.batch_size,
                                                   window_size=self.argv.window)

    @staticmethod
    def show_sample_stats(sample_set, vocab_label):
        pass


class RerankingPreprocessor(Preprocessor):

    def __init__(self, argv):
        super(RerankingPreprocessor, self).__init__(argv)

    def create_vocab_word(self, corpus):
        vocab_word = Vocab()
        vocab_word.set_init_word()
        vocab_word.add_vocab_from_lists(corpus=corpus, vocab_cut_off=self.argv.vocab_cut_off)
        vocab_word.add_word(UNK)
        say('\nVocab: %d\tType: word\n' % vocab_word.size())
        return vocab_word

    def set_sample_factory(self, vocab_word, vocab_label):
        self.sample_factory = RerankingSampleFactory(vocab_word=vocab_word,
                                                     vocab_label=vocab_label,
                                                     batch_size=self.argv.batch_size,
                                                     window_size=self.argv.window)


class GridPreprocessor(Preprocessor):

    def __init__(self, argv):
        super(GridPreprocessor, self).__init__(argv)

    def set_sample_factory(self, vocab_word, vocab_label):
        self.sample_factory = GridSampleFactory(vocab_word=vocab_word,
                                                vocab_label=vocab_label,
                                                batch_size=self.argv.batch_size,
                                                window_size=self.argv.window)
