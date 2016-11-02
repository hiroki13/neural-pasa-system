import numpy as np
from vocab import UNK, NA, GA, O, NI, PRD, GA_LABEL, O_LABEL, NI_LABEL


class Sample(object):

    def __init__(self, sent, window):
        """
        sent: 1D: n_words; Word
        word_ids: 1D: n_words; word id
        prd_indices: 1D: n_prds; word id
        x_w: 1D: n_prds, 2D: n_words, 3D: window; word id
        x_p: 1D: n_prds, 2D: n_words; posit phi id
        y: 1D: n_prds, 2D: n_words; label id
        """
        self.sent = sent
        self.word_ids = []
        self.label_ids = []
        self.word_phi = []
        self.posit_phi = []
        self.prd_indices = []

        self.n_words = len(sent)
        self.n_prds = 0
        self.window = window
        self.slide = window / 2

        self.x_w = []
        self.x_p = []
        self.y = []

    def set_params(self, vocab_word, vocab_label):
        self.set_word_ids(vocab_word)
        self.set_label_ids(vocab_label)
        word_phi = self.get_word_phi()
        posit_phi = self.get_posit_phi()
        self.set_x_y(word_phi, posit_phi)

    def set_word_ids(self, vocab_word):
        word_ids = []
        for w in self.sent:
            if w.form not in vocab_word.w2i:
                w_id = vocab_word.get_id(UNK)
            else:
                w_id = vocab_word.get_id(w.form)
            word_ids.append(w_id)
        self.word_ids = word_ids

    def set_label_ids(self, vocab_label):
        labels = []
        prd_indices = []

        for word in self.sent:
            if word.is_prd and word.has_args():
                label_seq = self.create_label_seq(prd=word, vocab_label=vocab_label)
                labels.append(label_seq)
                prd_indices.append(word.index)

        assert len(labels) == len(prd_indices)

        self.label_ids = labels
        self.prd_indices = prd_indices
        self.n_prds = len(prd_indices)

    def create_label_seq(self, prd, vocab_label):
        label_seq = [vocab_label.get_id(NA) for i in xrange(self.n_words)]
        label_seq[prd.index] = vocab_label.get_id(PRD)
        for case_label, arg_index in enumerate(prd.case_arg_index):
            if arg_index > -1:
                if case_label == GA_LABEL:
                    label_seq[arg_index] = vocab_label.get_id(GA)
                elif case_label == O_LABEL:
                    label_seq[arg_index] = vocab_label.get_id(O)
                else:
                    label_seq[arg_index] = vocab_label.get_id(NI)
        return label_seq

    def get_word_phi(self):
        phi = []

        ###################
        # Argument window #
        ###################
        window = self.window
        slide = self.slide
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

    def get_posit_phi(self):
        phi = []

        sent_len = len(self.word_ids)
        for prd_index in self.prd_indices:
            p_phi = [self.get_mark(prd_index, arg_index) for arg_index in xrange(sent_len)]
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)
        return phi

    def get_mark(self, prd_index, arg_index):
        slide = self.slide
        if prd_index - slide <= arg_index <= prd_index + slide:
            return 0
        return 1

    def set_x_y(self, word_phi, posit_phi):
        assert len(word_phi) == len(posit_phi) == len(self.label_ids)
        self.x_w = self._numpize(self._flatten(word_phi))
        self.x_p = self._numpize(self._flatten(posit_phi))
        self.y = self._numpize(self._flatten(self.label_ids))

    @staticmethod
    def _flatten(matrix):
        vec = []
        for row in matrix:
            vec.extend(row)
        return vec

    @staticmethod
    def _numpize(sample):
        return np.asarray(sample, dtype='int32')


class RankingSample(Sample):

    def __init__(self, sent, window):
        super(RankingSample, self).__init__(sent, window)

    def create_label_seq(self, prd, vocab_label):
        label_seq = []
        null_arg_index = self.n_words
        for index in prd.case_arg_index:
            if index < 0:
                case_arg_index = null_arg_index
            else:
                case_arg_index = index
            label_seq.append(case_arg_index)
        label_seq.append(prd.index)
        return label_seq

    def set_x_y(self, word_phi, posit_phi):
        assert len(word_phi) == len(posit_phi) == len(self.label_ids)
        self.x_w = self._numpize(self._flatten(word_phi))
        self.x_p = self._numpize(self._flatten(posit_phi))
        self.y = self._numpize(self.label_ids)


class RerankingSample(Sample):

    def __init__(self, n_best_list, window):
        super(RerankingSample, self).__init__(n_best_list.words, window)
        self.n_best_list = n_best_list
        self.x_l = []

    def set_params(self, vocab_word, vocab_label):
        self.set_word_ids(vocab_word)
        self.set_label_ids(vocab_label)
        word_phi = self.get_word_phi()
        posit_phi = self.get_posit_phi()
        self.set_x_y(word_phi, posit_phi)

    def set_label_ids(self, vocab_label):
        self.label_ids = self._get_max_f1_list_index(self.n_best_list)
        self.prd_indices = self.n_best_list.prd_indices
        self.n_prds = len(self.prd_indices)

    @staticmethod
    def _get_max_f1_list_index(n_best_list):
        return n_best_list.get_max_f1_list_index()

    def set_x_y(self, word_phi, posit_phi):
        assert len(word_phi) == len(posit_phi)
        self.x_w = self._numpize(word_phi)
        self.x_p = self._numpize(posit_phi)
        self.x_l = self._numpize(self.n_best_list.lists)
        self.y = self._numpize(self.label_ids)
