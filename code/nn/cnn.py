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


