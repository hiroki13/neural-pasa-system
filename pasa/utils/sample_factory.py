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
            if prev_n_words != n_words or index - prev_index > self.batch_size:
                batch_index.append((prev_index, index))
                num_prds.append(prev_n_prds)
                sent_length.append(prev_n_words)
                prev_index = index
                prev_n_prds = n_prds
                prev_n_words = n_words

            theano_x_w.extend(sample.x_w)
            theano_x_p.extend(sample.x_p)
            theano_y.extend(sample.y)
            index += len(sample.x_w)

        if index > prev_index:
            batch_index.append((prev_index, index))
            num_prds.append(prev_n_prds)
            sent_length.append(prev_n_words)

        assert len(batch_index) == len(sent_length) == len(num_prds)

        return [self._shared(theano_x_w),
                self._shared(theano_x_p),
                self._shared(theano_y),
                self._shared(sent_length),
                self._shared(num_prds)], batch_index

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

