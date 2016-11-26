import numpy as np

from abc import ABCMeta, abstractmethod
from sample import BaseSample, StackingSample, MixedPrdSample
from batch import BaseBatch, GridBatch


class SampleFactory(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv, vocab_word, vocab_label):
        self.argv = argv
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.batch_size = argv.batch_size
        self.window = argv.window

    @abstractmethod
    def create_samples(self, corpus):
        raise NotImplementedError

    @abstractmethod
    def create_sample(self, sent):
        raise NotImplementedError

    @abstractmethod
    def create_batch(self, samples):
        raise NotImplementedError


class BaseSampleFactory(SampleFactory):

    def create_samples(self, corpus):
        """
        :param corpus: 1D: n_docs * n_sents, 2D: n_words; elem=Word
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None
        return [self.create_sample(sent=sent) for sent in corpus]

    def create_sample(self, sent):
        return BaseSample(sent, self.window, self.vocab_word, self.vocab_label)

    def create_batch(self, samples):
        return BaseBatch(self.batch_size, samples)


class GridSampleFactory(BaseSampleFactory):

    def create_batch(self, samples):
        return GridBatch(self.batch_size, samples)


class StackingSampleFactory(BaseSampleFactory):

    def create_sample(self, sent):
        return StackingSample(sent=sent, window=self.window)

    def create_samples(self, corpus):
        """
        :param corpus: Results()
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None

        samples = []
        for sent in zip(corpus.samples, corpus.outputs_prob, corpus.outputs_hidden):
            sample = self.create_sample(sent=sent)
            sample.set_params(self.vocab_word, self.vocab_label)
            samples.append(sample)

        return samples

    def create_batch(self, samples):
        n_inputs = len(samples[0].inputs)
        batches = []
        batch = [[] for i in xrange(n_inputs)]

        samples = [sample for sample in samples if sample.n_prds > 0]
        samples = self._sort_by_n_words(samples)
        prev_n_prds = samples[0].n_prds
        prev_n_words = samples[0].n_words

        for sample in samples:
            if self.is_batch_boundary(sample.n_words, prev_n_words,
                                      sample.n_prds, prev_n_prds,
                                      len(batch[-1]), self.batch_size):
                prev_n_prds = sample.n_prds
                prev_n_words = sample.n_words
                batches.append(batch)
                batch = [[] for i in xrange(n_inputs)]

            batch = self._add_inputs_to_batch(batch, sample)

        if len(batch[0]) > 0:
            batches.append(batch)

        return batches

    @staticmethod
    def is_batch_boundary(n_words, prev_n_words, n_prds, prev_n_prds, n_batches, batch_size):
        if prev_n_words != n_words or n_prds != prev_n_prds or n_batches >= batch_size:
            return True
        return False

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: s.n_prds)
        samples.sort(key=lambda s: s.n_words)
        return samples

    @staticmethod
    def _add_inputs_to_batch(batch, sample):
        s = sample.sample
        batch[0].append(s.x_w)
        batch[1].append(s.x_p)
        batch[2].append(sample.x_w)
        batch[3].append(sample.x_p)
        batch[4].append(sample.y)
        return batch
