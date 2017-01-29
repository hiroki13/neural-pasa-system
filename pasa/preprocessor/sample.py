import numpy as np

from abc import ABCMeta, abstractmethod
from ..utils.io_utils import say
from ..ling.vocab import UNK, NA, GA, O, NI, PRD, GA_INDEX, O_INDEX, NI_INDEX


class Sample(object):
    __metaclass__ = ABCMeta

    def __init__(self, sent, mark_phi, window, vocab_word, vocab_label):
        """
        sent: 1D: n_words; Word()
        word_ids: 1D: n_words; word id
        prd_indices: 1D: n_prds; prd index
        x: 1D: n_elems
        y: 1D: n_prds, 2D: n_words; label id
        """
        self.sent = sent
        self.prd_indices = self._set_prd_indices(sent)
        self.n_words = len(sent)
        self.n_prds = len(self.prd_indices)

        self.word_ids = self._set_word_ids(sent, vocab_word)
        self.label_ids = self._set_label_ids(sent, vocab_label)

        self.x = self._set_x(mark_phi=mark_phi, window=window)
        self.y = self._set_y()

    @abstractmethod
    def _set_word_ids(self, sent, vocab_word):
        raise NotImplementedError

    @abstractmethod
    def _set_label_ids(self, sent, vocab_label):
        raise NotImplementedError

    @abstractmethod
    def _set_prd_indices(self, sent):
        raise NotImplementedError

    @abstractmethod
    def _set_x(self, mark_phi, window):
        raise NotImplementedError

    @abstractmethod
    def _set_y(self):
        raise NotImplementedError

    @staticmethod
    def _numpize(sample):
        return np.asarray(sample, dtype='int32')


class BaseSample(Sample):

    def _set_word_ids(self, sent, vocab_word):
        word_ids = []
        for w in sent:
            if w.form not in vocab_word.w2i:
                w_id = vocab_word.get_id(UNK)
            else:
                w_id = vocab_word.get_id(w.form)
            word_ids.append(w_id)
        return word_ids

    def _set_label_ids(self, sent, vocab_label):
        labels = []
        for word in sent:
            if word.is_prd and word.has_args():
                label_seq = self._create_label_seq(prd=word, n_words=self.n_words, vocab_label=vocab_label)
                labels.append(label_seq)
        return labels

    def _set_prd_indices(self, sent):
        return [word.index for word in sent if word.is_prd and word.has_args()]

    @staticmethod
    def _create_label_seq(prd, n_words, vocab_label):
        label_seq = [vocab_label.get_id(NA) for i in xrange(n_words)]
        label_seq[prd.index] = vocab_label.get_id(PRD)
        for case_index, arg_index in enumerate(prd.arg_indices):
            if arg_index > -1:
                if case_index == GA_INDEX:
                    label_seq[arg_index] = vocab_label.get_id(GA)
                elif case_index == O_INDEX:
                    label_seq[arg_index] = vocab_label.get_id(O)
                elif case_index == NI_INDEX:
                    label_seq[arg_index] = vocab_label.get_id(NI)
                else:
                    say('\nSomething wrong with case labels\n')
                    exit()
        return label_seq

    def _set_x(self, mark_phi, window):
        x = []
        x.append(self._get_word_phi(window))
        if mark_phi:
            x.append(self._get_posit_phi())
        return x

    def _set_y(self):
        return self._numpize(self.label_ids)

    def _get_word_phi(self, window):
        phi = []

        slide = window / 2
        pad = [0 for i in xrange(slide)]
        p_sent_w_ids = pad + self.word_ids + pad
        a_sent_w_ids = self.word_ids

        for prd_index in self.prd_indices:
            prd_ctx = p_sent_w_ids[prd_index: prd_index + window]
            p_phi = []

            for arg_index in xrange(self.n_words):
                arg_ctx = [a_sent_w_ids[arg_index]] + prd_ctx
                p_phi.append(arg_ctx)
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)
        return self._numpize(phi)

    def _get_posit_phi(self):
        phi = self._create_rel_phi()
        assert len(phi) == len(self.prd_indices)
        return self._numpize(phi)

    def _create_rel_phi(self):
        phi = []
        for i, p_index in enumerate(self.prd_indices):
            tmp_phi = [self._get_prd_flag(p_index, a_index) for a_index in xrange(self.n_words)]
            phi.append(tmp_phi)
        return phi

    @staticmethod
    def _get_prd_flag(w1, w2):
        if w1 == w2:
            return 1
        return 0

