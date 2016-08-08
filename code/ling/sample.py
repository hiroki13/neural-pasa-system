import numpy as np
from vocab import UNK, NA, GA, O, NI, PRD, GA_LABEL, O_LABEL


class Sample(object):

    def __init__(self, sent, window):
        """
        sent: 1D: n_words; Word
        word_ids: 1D: n_words; word_id
        prd_indices: 1D: n_prds; word_id
        x: 1D: n_prds, 2D: n_words, 3D: window; word_id
        y: 1D: n_prds, 2D: n_words; label_id
        """
        self.sent = sent
        self.word_ids = []
        self.label_ids = []
        self.phi = []
        self.prd_indices = []

        self.n_words = len(sent)
        self.n_prds = 0
        self.window = window
        self.slide = window / 2

        self.x = []
        self.y = []

    def set_params(self, vocab_word, vocab_label):
        self.set_word_ids(vocab_word)
        self.set_label_ids(vocab_label)
        self.set_phi()
        self.set_x_y()

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
            # check if the word is a predicate or not
            if word.is_prd:
                p_labels = [vocab_label.get_id(NA) for i in xrange(self.n_words)]
                p_labels[word.index] = vocab_label.get_id(PRD)

                is_arg = False
                for case_label, arg_index in enumerate(word.case_arg_index):
                    if arg_index > -1:
                        if case_label == GA_LABEL:
                            p_labels[arg_index] = vocab_label.get_id(GA)
                        elif case_label == O_LABEL:
                            p_labels[arg_index] = vocab_label.get_id(O)
                        else:
                            p_labels[arg_index] = vocab_label.get_id(NI)
                        is_arg = True

                if is_arg:
                    labels.append(p_labels)
                    prd_indices.append(word.index)

        assert len(labels) == len(prd_indices)

        self.label_ids = labels
        self.prd_indices = prd_indices
        self.n_prds = len(prd_indices)

    def set_phi(self):
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
                arg_ctx.append(self.get_mark(prd_index, arg_index))
                p_phi.append(arg_ctx)
            phi.append(p_phi)

        assert len(phi) == len(self.prd_indices)

        self.phi = phi

    def get_mark(self, prd_index, arg_index):
        slide = self.slide
        if prd_index - slide <= arg_index <= prd_index + slide:
            return 1
        return 2

    def set_x_y(self):
        x = []
        y = []
        for sent_phi, sent_label in zip(self.phi, self.label_ids):
            x += sent_phi
            y += sent_label
        self.x = numpize(x)
        self.y = numpize(y)


def numpize(sample):
    return np.asarray(sample, dtype='int32')
