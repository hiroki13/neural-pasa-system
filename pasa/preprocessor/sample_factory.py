import numpy as np

from abc import ABCMeta, abstractmethod
from sample import Sample, StackingSample, RerankingSample, GridSample


class SampleFactory(object):
    __metaclass__ = ABCMeta

    def __init__(self, vocab_word, vocab_label, batch_size, window_size):
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.batch_size = batch_size
        self.window = window_size

    @abstractmethod
    def create_sample(self, sent):
        raise NotImplementedError()

    def create_samples(self, corpus):
        """
        :param corpus: 1D: n_docs * n_sents, 2D: n_words; elem=Word
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None

        samples = []
        for sent in corpus:
            sample = self.create_sample(sent=sent)
            sample.set_params(self.vocab_word, self.vocab_label)
            samples.append(sample)

        return samples

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: len(s[0]))
        return samples

    def create_batched_samples(self, samples, n_inputs):
        raise NotImplementedError()

    @staticmethod
    def _add_inputs_to_batch(batch, sample):
        raise NotImplementedError()

    @staticmethod
    def _is_batch_boundary(boundary_elems, batch_size):
        raise NotImplementedError()

    @staticmethod
    def separate_samples(samples):
        return [elem for sample in samples for elem in zip(*sample.inputs)]


class BasicSampleFactory(SampleFactory):

    def create_sample(self, sent):
        return Sample(sent=sent, window=self.window)

    def create_batched_samples(self, samples, n_inputs):
        """
        :param samples: 1D: n_sents; Sample
        """
        batches = []
        batch = [[] for i in xrange(n_inputs)]

        samples = [sample for sample in samples if sample.n_prds > 0]
        samples = self.separate_samples(samples)
        samples = self._sort_by_n_words(samples)
        prev_n_words = len(samples[0][0])

        for sample in samples:
            n_words = len(sample[0])
            boundary_elems = (n_words, prev_n_words, len(batch[-1]))

            if self._is_batch_boundary(boundary_elems, self.batch_size):
                prev_n_words = n_words
                batches.append(batch)
                batch = [[] for i in xrange(n_inputs)]
            batch = self._add_inputs_to_batch(batch, sample)

        if len(batch[0]) > 0:
            batches.append(batch)

        return batches

    @staticmethod
    def _is_batch_boundary(boundary_elems, batch_size):
        n_words, prev_n_words, n_batches = boundary_elems
        if prev_n_words != n_words or n_batches >= batch_size:
            return True
        return False

    @staticmethod
    def _add_inputs_to_batch(batch, sample):
        batch[0].append(sample[0])
        batch[1].append(sample[1])
        batch[2].append(sample[2])
        return batch


class StackingSampleFactory(BasicSampleFactory):

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

    def create_batched_samples(self, samples, n_inputs):
        """
        :param samples: 1D: n_sents; Sample
        """
        batches = []
        batch = [[] for i in xrange(n_inputs)]

        samples = [sample for sample in samples if sample.n_prds > 0]
        samples = self._sort_by_n_words(samples)
        prev_n_prds = samples[0].n_prds
        prev_n_words = samples[0].n_words

        for sample in samples:
            if self.is_batch_boundary(sample.n_words, prev_n_words,
                                      sample.n_prds, prev_n_prds,
                                      len(batch[2]), self.batch_size):
                prev_n_prds = sample.n_prds
                prev_n_words = sample.n_words
                batches.append(batch)
                batch = [[] for i in xrange(n_inputs)]

            batch[0].append(sample.x_w)
            batch[1].append(sample.x_p)
            batch[2].append(sample.y)

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
        batch[0].append(sample[0])
        batch[1].append(sample[1])
        batch[2].append(sample[2])
        return batch


class RerankingSampleFactory(SampleFactory):

    def create_sample(self, sent):
        return RerankingSample(n_best_list=sent, window=self.window)

    def create_samples(self, corpus, test=False):
        """
        :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
        :param test: whether the corpus is for dev or test
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None

        # 1D: n_docs * n_sents, 2D: n_words; elem=Word
        if test is False:
            corpus = self._sort_by_n_words(corpus)

        samples = []
        for n_best_list in corpus:
            if len(n_best_list.lists) == 0:
                continue
            sample = self.create_sample(sent=n_best_list)
            sample.set_params(self.vocab_word, self.vocab_label)
            samples.append(sample)

        return samples

    def create_shared_batch_samples(self, samples):
        """
        :param samples: 1D: n_sents; Sample
        """
        n_elems = 5
        batches = []
        batch = [[] for i in xrange(n_elems)]

        samples = [sample for sample in samples if sample.n_prds > 0]
        prev_n_prds = samples[0].n_prds
        prev_n_words = samples[0].n_words

        for sample in samples:
            if self.is_batch_boundary(sample.n_words, prev_n_words,
                                      sample.n_prds, prev_n_prds,
                                      len(batch[3]), self.batch_size):
                prev_n_prds = sample.n_prds
                prev_n_words = sample.n_words
                batches.append(batch)
                batch = [[] for i in xrange(n_elems)]

            batch[0].append(sample.x_w)
            batch[1].append(sample.x_p)
            batch[2].append(sample.x_l)
            batch[3].append(sample.y)
            batch[4].extend(sample.label_ids)

        if len(batch[0]) > 0:
            batches.append(batch)

        return batches

    @staticmethod
    def is_batch_boundary(n_words, prev_n_words, n_prds, prev_n_prds, n_batches, batch_size):
        if prev_n_words != n_words or n_prds != prev_n_prds or n_batches == batch_size:
            return True
        return False

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: len(s.words))
        return samples


class GridSampleFactory(BasicSampleFactory):

    def create_sample(self, sent):
        return GridSample(sent=sent, window=self.window)

    def create_batched_samples(self, samples, n_inputs=3):
        """
        :param samples: 1D: n_sents; Sample
        """
        batches = []
        batch = [[] for i in xrange(n_inputs)]

        samples = [sample for sample in samples if sample.n_prds > 0]
        samples = self._sort_by_n_words(samples)
        prev_n_prds = samples[0].n_prds
        prev_n_words = samples[0].n_words

        for sample in samples:
            if self.is_batch_boundary(sample.n_words, prev_n_words,
                                      sample.n_prds, prev_n_prds,
                                      len(batch[2]), self.batch_size):
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
    def _add_inputs_to_batch(batch, sample):
        batch[0].append(sample.x_w)
        batch[1].append(sample.x_p)
        batch[2].append(sample.y)
        return batch

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: s.n_prds)
        samples.sort(key=lambda s: s.n_words)
        return samples
