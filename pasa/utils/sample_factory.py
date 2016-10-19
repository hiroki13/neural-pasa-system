import numpy as np
import theano

from abc import ABCMeta, abstractmethod
from ..ling.sample import Sample, RankingSample


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

    def create_samples(self, corpus, test=False):
        """
        :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
        :param test: whether the corpus is for dev or test
        :return: samples: 1D: n_samples; Sample
        """
        if corpus is None:
            return None

        # 1D: n_docs * n_sents, 2D: n_words; elem=Word
        corpus = [sent for doc in corpus for sent in doc]
        if test is False:
            corpus = self._sort_by_n_words(corpus)

        samples = []
        for sent in corpus:
            sample = self.create_sample(sent=sent)
            sample.set_params(self.vocab_word, self.vocab_label)
            samples.append(sample)

        return samples

    def create_shared_batch_samples(self, samples):
        """
        :param samples: 1D: n_sents; Sample
        """
        theano_x_w = []
        theano_x_p = []
        theano_y = []
        sent_len = []
        n_prds = []
        batch_index = []

        samples = [sample for sample in samples if sample.n_prds > 0]
        prev_n_prds = samples[0].n_prds
        prev_n_words = samples[0].n_words
        prev_index = 0
        index = 0

        for sample in samples:
            if self.is_batch_boundary(sample.n_words,
                                      prev_n_words,
                                      index,
                                      prev_index,
                                      self.batch_size):
                batch_index.append((prev_index, index))
                n_prds.append(prev_n_prds)
                sent_len.append(prev_n_words)
                prev_index = index
                prev_n_prds = sample.n_prds
                prev_n_words = sample.n_words

            theano_x_w.extend(sample.x_w)
            theano_x_p.extend(sample.x_p)
            theano_y.extend(sample.y)
            index += len(sample.x_w)

        if index > prev_index:
            batch_index.append((prev_index, index))
            n_prds.append(prev_n_prds)
            sent_len.append(prev_n_words)

        assert len(batch_index) == len(sent_len) == len(n_prds)

        return [self._shared(theano_x_w),
                self._shared(theano_x_p),
                self._shared(theano_y),
                self._shared(sent_len),
                self._shared(n_prds)], batch_index

    @staticmethod
    def is_batch_boundary(n_words, prev_n_words, index, prev_index, batch_size):
        if prev_n_words != n_words or index - prev_index > batch_size:
            return True
        return False

    @staticmethod
    def _sort_by_n_words(corpus):
        np.random.shuffle(corpus)
        corpus.sort(key=lambda s: len(s))
        return corpus

    @staticmethod
    def _shared(matrix):
        return theano.shared(np.asarray(matrix, dtype='int32'), borrow=True)


class BasicSampleFactory(SampleFactory):

    def create_sample(self, sent):
        return Sample(sent=sent, window=self.window)


class RankingSampleFactory(SampleFactory):

    def create_sample(self, sent):
        return RankingSample(sent=sent, window=self.window)

    def create_shared_batch_samples(self, samples):
        """
        :param samples: 1D: n_sents; Sample
        """
        batches = []
        x_w = []
        x_p = []
        y = []

        samples = [sample for sample in samples if sample.n_prds > 0]
        prev_n_words = samples[0].n_words

        for sample in samples:
            if self.is_batch_boundary(sample.n_words,
                                      prev_n_words,
                                      y,
                                      self.batch_size):

                batches.append(self.create_batch(x_w, x_p, y, prev_n_words))
                prev_n_words = sample.n_words
                x_w = []
                x_p = []
                y = []

            x_w.extend(sample.x_w)  # 1D: n_prds * n_words, 2D: 5 + window
            x_p.extend(sample.x_p)  # 1D: n_prds * n_words
            y.extend(sample.y)  # 1D: n_prds, 2D: n_labels (3)

        if x_w:
            batches.append(self.create_batch(x_w, x_p, y, prev_n_words))

        return batches

    def create_batch(self, x_w, x_p, y, n_words):
        return self._numpize(x_w), self._numpize(x_p), self._numpize(y), self._numpize(n_words)

    @staticmethod
    def _numpize(sample):
        return np.asarray(sample, dtype='int32')

    @staticmethod
    def is_batch_boundary(n_words, prev_n_words, y, batch_size):
        if prev_n_words != n_words or len(y) >= batch_size:
            return True
        return False
