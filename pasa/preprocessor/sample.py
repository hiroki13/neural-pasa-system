import numpy as np

from abc import ABCMeta, abstractmethod
from ..utils.io_utils import say
from ..ling.vocab import UNK, NA, GA, O, NI, PRD, GA_LABEL, O_LABEL, NI_LABEL


class Sample(object):
    __metaclass__ = ABCMeta

    def __init__(self, sent, window, vocab_word, vocab_label):
        """
        sent: 1D: n_words; Word()
        word_ids: 1D: n_words; word id
        prd_indices: 1D: n_prds; prd index
        x: 1D: n_elems
        y: 1D: n_prds, 2D: n_words; label id
        """
        self.sent = sent
        self.n_words = len(sent)
        self.prd_indices = self._set_prd_indices(sent)
        self.n_prds = len(self.prd_indices)

        self.word_ids = self._set_word_ids(sent, vocab_word)
        self.label_ids = self._set_label_ids(sent, vocab_label)

        self.x = self._set_x(window)
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
    def _set_x(self, window):
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
        for case_label, arg_index in enumerate(prd.case_arg_index):
            if arg_index > -1:
                if case_label == GA_LABEL:
                    label_seq[arg_index] = vocab_label.get_id(GA)
                elif case_label == O_LABEL:
                    label_seq[arg_index] = vocab_label.get_id(O)
                elif case_label == NI_LABEL:
                    label_seq[arg_index] = vocab_label.get_id(NI)
                else:
                    say('\nSomething wrong with case labels\n')
                    exit()
        return label_seq

    def _set_x(self, window):
        x_w = self._numpize(self._get_word_phi(window))
        x_p = self._numpize(self._get_posit_phi(window))
        return [x_w, x_p]

    def _set_y(self):
        return self._numpize(self.label_ids)

    def _get_word_phi(self, window):
        phi = []

        ###################
        # Argument window #
        ###################
        slide = window / 2
        sent_len = len(self.word_ids)
        pad = [0 for i in xrange(slide)]
        a_sent_w_ids = pad + self.word_ids + pad

        ####################
        # Predicate window #
        ####################
        p_window = 5
        p_slide = p_window / 2
        p_pad = [0 for i in xrange(p_slide)]
        p_sent_w_ids = p_pad + self.word_ids + p_pad

        for prd_index in self.prd_indices:
            prd_ctx = p_sent_w_ids[prd_index: prd_index + p_window]
            p_phi = []

            for arg_index in xrange(sent_len):
                arg_ctx = a_sent_w_ids[arg_index: arg_index + window] + prd_ctx
                p_phi.append(arg_ctx)
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)
        return phi

    def _get_posit_phi(self, window):
        phi = []
        slide = window / 2

        for prd_index in self.prd_indices:
#            p_phi = [self._get_mark(prd_index, arg_index, slide) for arg_index in xrange(self.n_words)]
            p_phi = [self._get_relative_posit(prd_index, arg_index) for arg_index in xrange(self.n_words)]
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)
        return phi

    @staticmethod
    def _get_mark(prd_index, arg_index, slide):
        if prd_index - slide <= arg_index <= prd_index + slide:
            return 0
        return 1

    @staticmethod
    def _get_relative_posit(w1, w2):
        dist = w2 - w1
        if -5 <= dist < 0:
            return -1 * dist + 5
        elif 5 < dist < 10:
            return 11
        elif -10 < dist < -5:
            return 12
        elif dist <= -10:
            return 13
        elif dist >= 10:
            return 14
        return dist


class MentionPairSample(Sample):

    def _set_word_ids(self, doc, vocab_word):
        word_ids = []
        for sent in doc:
            tmp_word_ids = []
            for w in sent:
                if w.form not in vocab_word.w2i:
                    w_id = vocab_word.get_id(UNK)
                else:
                    w_id = vocab_word.get_id(w.form)
                tmp_word_ids.append(w_id)
            word_ids.append(tmp_word_ids)
        return word_ids

    def _set_label_ids(self, doc, vocab_label):
        labels = []
        for sent in doc:
            for word in sent:
                if word.is_prd:
                    for case_index, indices in enumerate(word.inter_case_arg_index):
                        label = self._create_label(prd=word, doc=doc, case_index=case_index, indices=indices)
                        labels.extend(label)
        return labels

    @staticmethod
    def _create_label(prd, doc, case_index, indices):
        p_labels = []
        n_labels = []
        p_index = (prd.sent_index, prd.index)
        for sent in doc:
            for word in sent:
                a_index = (word.sent_index, word.index)
                if a_index in indices:
                    p_labels.append((p_index, a_index, case_index, 1))
                else:
                    n_labels.append((p_index, a_index, case_index, 0))
        return p_labels + n_labels[-1:]

    def _set_prd_indices(self, doc):
        return [(w.sent_index, w.index) for sent in doc for w in sent if w.is_prd]

    def _set_x(self, window):
        return self._numpize(self.extract_phi(self.label_ids))

    def extract_phi(self, label_sets):
        return [self.extract_phi_each(label_set, self.word_ids) for label_set in label_sets]

    @staticmethod
    def extract_phi_each(label_set, doc):
        p_index, a_index, case_index, label = label_set
        p_sent_index, p_word_index = p_index
        a_sent_index, a_word_index = a_index
        p_id = doc[p_sent_index][p_word_index]
        a_id = doc[a_sent_index][a_word_index]
        return [p_id, a_id]

    def _set_y(self):
        return self._numpize(self.extract_label(self.label_ids))

    @staticmethod
    def extract_label(label_sets):
        return [label_set[-1] for label_set in label_sets]
