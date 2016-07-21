from utils import sample_weights, tanh, relu

import theano
import theano.tensor as T


def layers(x, window, dim_emb, dim_hidden, n_layers, activation=tanh):
    params = []
    zero = T.zeros((1, dim_emb * window), dtype=theano.config.floatX)

    def zero_pad_gate(matrix):
        return T.neq(T.sum(T.eq(matrix, zero), 1, keepdims=True), dim_emb * window)

    for i in xrange(n_layers):
        if i == 0:
            W = theano.shared(sample_weights(dim_emb * window, dim_hidden))
#            h = zero_pad_gate(x) * relu(T.dot(x, W))
            h = relu(T.dot(x, W))
        else:
            W = theano.shared(sample_weights(dim_hidden, dim_hidden))
            h = activation(T.dot(h, W))
        params.append(W)

    return h, params


class Layer(object):

    def __init__(self, n_h):
        self.W = theano.shared(sample_weights(n_h, n_h))
        self.W_m = theano.shared(sample_weights(n_h * 2, n_h))
        self.params = [self.W, self.W_m]

    def convolution(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D n_h
        :return: y: 1D: n_words, 2D: batch_size, 3D: n_h
        """
        h_multi = h.reshape((h.shape[0], h.shape[1] / n_prds, n_prds, h.shape[2]))
        # 1D: n_words, 2D: batch_size, 3D: n_h
        h_multi = T.repeat(T.max(tanh(T.dot(h_multi, self.W)), axis=2), n_prds, 1)
        return tanh(T.dot(T.concatenate([h, h_multi], axis=2), self.W_m))

