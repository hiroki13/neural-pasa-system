from abc import ABCMeta, abstractmethod
from sample import BaseSample
from batch import BaseBatch, GridBatch


class SampleFactory(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv, vocab_word, vocab_label):
        self.argv = argv
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.batch_size = argv.batch_size

    @abstractmethod
    def create_samples(self, corpus):
        raise NotImplementedError

    @abstractmethod
    def _create_sample(self, sent):
        raise NotImplementedError

    @abstractmethod
    def create_batches(self, samples):
        raise NotImplementedError


class BaseSampleFactory(SampleFactory):

    def create_samples(self, corpus):
        """
        :param corpus: 1D: n_docs * n_sents, 2D: n_words; elem=Word
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None
        return [self._create_sample(sent) for sent in corpus]

    def _create_sample(self, sent):
        return BaseSample(sent, self.argv.mark_phi, self.argv.window, self.vocab_word, self.vocab_label)

    def create_batches(self, samples):
        return BaseBatch(self.batch_size, samples)


class GridSampleFactory(BaseSampleFactory):

    def create_batches(self, samples):
        return GridBatch(self.batch_size, samples)
