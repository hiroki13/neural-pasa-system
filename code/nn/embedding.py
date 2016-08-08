import theano
import theano.tensor as T

from utils import sample_weights, build_shared_zeros


class EmbeddingLayer(object):
    def __init__(self, n_vocab, dim_emb, init_emb=None):
        self.E = None
        self.emb = None
        self.params = []
        self.set_layer(n_vocab, dim_emb, init_emb)

    def set_layer(self, n_vocab, dim_emb, init_emb):
        pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
        else:
            self.emb = theano.shared(init_emb)

        self.E = T.concatenate([pad, self.emb], 0)
        self.params.append(self.emb)

    def lookup(self, x):
        return self.E[x]

