import theano
import theano.tensor as T

from nn_utils import sample_weights, build_shared_zeros


class EmbeddingLayer(object):

    def __init__(self, init_emb, n_vocab, dim_emb, n_posit, dim_posit, fix=0):
        self.E = None
        self.word_emb = None
        self.posit_emb = None
        self.params = []
        self.set_layer(init_emb, n_vocab, dim_emb, n_posit, dim_posit, fix)

    def set_layer(self, init_emb, n_vocab, dim_emb, n_posit, dim_posit, fix):
        self.word_emb = self.create_word_emb(init_emb, n_vocab, dim_emb)
        self.posit_emb = self.create_posit_emb(n_posit, dim_posit)

        if fix:
            self.params.extend([self.posit_emb])
        else:
            self.params.extend([self.word_emb, self.posit_emb])

        pad = build_shared_zeros((1, dim_emb))
        self.E = T.concatenate([pad, self.word_emb], 0)

    @staticmethod
    def create_word_emb(init_emb, n_vocab, dim_emb):
        if init_emb is None:
            return theano.shared(sample_weights(n_vocab - 1, dim_emb))
        return theano.shared(init_emb)

    @staticmethod
    def create_posit_emb(n_posit, dim_posit):
        return theano.shared(sample_weights(n_posit, dim_posit))

    def forward_word(self, x):
        return self.E[x]

    def forward_posit(self, x):
        return self.posit_emb[x]


class RerankingEmbeddingLayer(object):

    def __init__(self, init_emb, n_vocab, dim_emb, n_posit, dim_posit, n_labels, dim_label, fix=0):
        self.E = None
        self.word_emb = None
        self.posit_emb = None
        self.label_emb = None
        self.params = []
        self.set_layer(init_emb, n_vocab, dim_emb, n_posit, dim_posit, n_labels, dim_label, fix)

    def set_layer(self, init_emb, n_vocab, dim_emb, n_posit, dim_posit, n_labels, dim_label, fix):
        self.word_emb = self.create_word_emb(n_vocab, init_emb, dim_emb)
        self.posit_emb = self.create_emb(n_posit, dim_posit)
        self.label_emb = self.create_emb(n_labels, dim_label)

        if fix:
            self.params.extend([self.posit_emb, self.label_emb])
        else:
            self.params.extend([self.word_emb, self.posit_emb, self.label_emb])

        pad = build_shared_zeros((1, dim_emb))
        self.E = T.concatenate([pad, self.word_emb], 0)

    @staticmethod
    def create_word_emb(n_vocab, init_emb, dim_emb):
        if init_emb is None:
            return theano.shared(sample_weights(n_vocab - 1, dim_emb))
        return theano.shared(init_emb)

    @staticmethod
    def create_emb(dim_row, dim_column):
        return theano.shared(sample_weights(dim_row, dim_column))

    def forward_word(self, x):
        return self.E[x]

    def forward_posit(self, x):
        return self.posit_emb[x]

    def forward_label(self, x):
        return self.label_emb[x]

