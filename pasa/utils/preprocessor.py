from ..ling.vocab import Vocab, UNK
from ..utils.sample_factory import BasicSampleFactory, RankingSampleFactory
from io_utils import CorpusLoader, say, dump_data, load_data, load_init_emb
from stats import corpus_statistics, sample_statistics


class Preprocessor(object):

    def __init__(self, argv):
        self.argv = argv
        self.corpus_loader = None
        self.sample_factory = None

    def set_corpus_loader(self):
        self.corpus_loader = CorpusLoader(min_unit='word', data_size=self.argv.data_size)

    def set_sample_factory(self, vocab_word, vocab_label):
        self.sample_factory = BasicSampleFactory(vocab_word=vocab_word,
                                                 vocab_label=vocab_label,
                                                 batch_size=self.argv.batch_size,
                                                 window_size=self.argv.window)

    def load_corpus_set(self):
        # corpus: 1D: n_sents, 2D: n_words, 3D: Word()
        train_corpus = self.corpus_loader.load_corpus(path=self.argv.train_data)
        dev_corpus = self.corpus_loader.load_corpus(path=self.argv.dev_data)
        test_corpus = self.corpus_loader.load_corpus(path=self.argv.test_data)
        return train_corpus, dev_corpus, test_corpus

    def create_sample_set(self, corpus_set):
        # samples: 1D: n_sents; Sample
        train_corpus, dev_corpus, test_corpus = corpus_set
        train_samples = self.create_samples(train_corpus, test=False)
        dev_samples = self.create_samples(dev_corpus, test=True)
        test_samples = self.create_samples(test_corpus, test=True)
        return train_samples, dev_samples, test_samples

    def create_samples(self, corpus, test):
        return self.sample_factory.create_samples(corpus, test)

    def create_shared_samples(self, samples):
        return self.sample_factory.create_shared_batch_samples(samples)

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
        vocab_word.add_word(UNK)
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
