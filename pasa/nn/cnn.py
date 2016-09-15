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

    def __init__(self, n_h, pooling, k=2):
        self.k = k
#        self.W_m = theano.shared(sample_weights(n_h, n_h))
        self.W_m1 = theano.shared(sample_weights(n_h, n_h))
        self.W_m2 = theano.shared(sample_weights(n_h, n_h))
        self.W_m3 = theano.shared(sample_weights(n_h, n_h))
#        self.W_c = theano.shared(sample_weights(n_h * (k+1), n_h))
        self.W_c = theano.shared(sample_weights(n_h * 2, n_h))
#        self.W_k = theano.shared(sample_weights(n_h, n_h))
        self.pooling = pooling

#        self.params = [self.W_m, self.W_c]
        self.params = [self.W_m1, self.W_m2, self.W_m3, self.W_c]
#        self.params = [self.W_m, self.W_c, self.W_k]

    def convolution2(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size * n_prds, 3D n_h
        :return: 1D: n_words, 2D: batch_size * n_prds, 3D: n_h
        """

        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        m = h.reshape((h.shape[0], h.shape[1] / n_prds, n_prds, h.shape[2]))
        # 1D: n_words, 2D: batch_size, 3D: n_h
        h_m = self.pooling(relu(T.dot(m, self.W_m)), axis=2)
        # 1D: n_words, 2D: batch_size * n_prds, 3D: n_h
        h_m = T.repeat(h_m, n_prds, 1)

        return relu(T.dot(T.concatenate([h, h_m], axis=2), self.W_c))

    def convolution(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size * n_prds, 3D n_h
        :return: 1D: n_words, 2D: batch_size * n_prds, 3D: n_h
        """

        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        m = h.reshape((h.shape[0], h.shape[1] / n_prds, n_prds, h.shape[2]))
        m = m.dimshuffle((1, 2, 0, 3))
#        n_words = m.shape[2]

        # 1D: batch_size, 2D: n_prds * n_words, 3D: n_h
        m1 = m.reshape((m.shape[0], m.shape[1] * m.shape[2], m.shape[3]))
        m3 = m.dimshuffle((0, 2, 1, 3))

        # 1D: batch_size, 2D: n_h
        h_m1 = T.max(relu(T.dot(m1, self.W_m1)), axis=1)
        h_m1 = h_m1.dimshuffle((0, 'x', 'x', 1))
#        h_m1 = T.repeat(h_m1, n_prds, 1)
#        h_m1 = T.repeat(h_m1, n_words, 2)

        # 1D: batch_size, 2D: n_prds, 3D: n_h
        h_m2 = T.max(relu(T.dot(m, self.W_m2)), axis=2)
        h_m2 = h_m2.dimshuffle((0, 1, 'x', 2))
#        h_m2 = T.repeat(h_m2, n_words, 2)

        # 1D: batch_size, 2D: n_words, 3D: n_h
        h_m3 = T.max(relu(T.dot(m3, self.W_m3)), axis=2)
        h_m3 = h_m3.dimshuffle((0, 'x', 1, 2))
#        h_m3 = T.repeat(h_m3, n_prds, 1)

        # 1D: batch_size, 2D: n_prds, 3D: n_words, 4D: n_h
        h_m = h_m1 + h_m2 + h_m3
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        h_m = h_m.dimshuffle((2, 0, 1, 3))
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        h_m = h_m.reshape((h_m.shape[0], h_m.shape[1] * h_m.shape[2], h_m.shape[3]))

        return relu(T.dot(T.concatenate([h, h_m], axis=2), self.W_c))

    def convolution_max(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size * n_prds, 3D n_h
        :return: 1D: n_words, 2D: batch_size * n_prds, 3D: n_h
        """

        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        m = h.reshape((h.shape[0], h.shape[1] / n_prds, n_prds, h.shape[2]))
        # 1D: batch_size, 2D: n_prds * n_words, 3D: n_h
        m = m.dimshuffle((1, 2, 0, 3))
        m = m.reshape((m.shape[0], m.shape[1] * m.shape[2], m.shape[3]))

        # 1D: batch_size, 2D: n_h
        h_m = T.max(relu(T.dot(m, self.W_m)), axis=1)
        # 1D: batch_size, 2D: n_h
        h_m = h_m.reshape((h_m.shape[0], -1)).dimshuffle((0, 'x', 'x', 1))
        # 1D: batch_size, 2D: n_prds, 3D: n_h
        h_m = T.repeat(h_m, n_prds, 1)
        # 1D: batch_size, 2D: n_prds, 3D: n_words, 4D: n_h
        h_m = T.repeat(h_m, h.shape[0], 2)
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        h_m = h_m.dimshuffle((2, 0, 1, 3))
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        h_m = h_m.reshape((h_m.shape[0], h_m.shape[1] * h_m.shape[2], h_m.shape[3]))

        return relu(T.dot(T.concatenate([h, h_m], axis=2), self.W_c))

    def convolution_k(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size * n_prds, 3D n_h
        :return: 1D: n_words, 2D: batch_size * n_prds, 3D: n_h
        """

        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        m = h.reshape((h.shape[0], h.shape[1] / n_prds, n_prds, h.shape[2]))
        # 1D: batch_size, 2D: n_prds * n_words, 3D: n_h
        m = m.dimshuffle((1, 2, 0, 3))
        m = m.reshape((m.shape[0], m.shape[1] * m.shape[2], m.shape[3]))

        # 1D: batch_size, 2D: k, 3D: n_h
#        h_m = k_max_pooling(relu(T.dot(m, self.W_m)), k=n_prds, axis=2)
        h_m = T.max(relu(T.dot(m, self.W_m)), axis=1)
        # 1D: batch_size, 2D: n_h
#        h_m = T.mean(T.dot(h_m, self.W_k), axis=1)

        # 1D: batch_size, 2D: k * n_h
        h_m = h_m.reshape((h_m.shape[0], -1)).dimshuffle((0, 'x', 'x', 1))
        # 1D: batch_size, 2D: n_prds, 3D: k * n_h
        h_m = T.repeat(h_m, n_prds, 1)
        # 1D: batch_size, 2D: n_prds, 3D: n_words, 4D: k * n_h
        h_m = T.repeat(h_m, h.shape[0], 2)
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: k * n_h
        h_m = h_m.dimshuffle((2, 0, 1, 3))
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: k * n_h
        h_m = h_m.reshape((h_m.shape[0], h_m.shape[1] * h_m.shape[2], h_m.shape[3]))

        return relu(T.dot(T.concatenate([h, h_m], axis=2), self.W_c))

    def convolution3(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size * n_prds, 3D n_h
        :return: 1D: n_words, 2D: batch_size * n_prds, 3D: n_h
        """

        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        m = h.reshape((h.shape[0], h.shape[1] / n_prds, n_prds, h.shape[2]))
        # 1D: batch_size, 2D: n_prds * n_words, 3D: n_h
        m = m.dimshuffle((1, 2, 0, 3))
        m = m.reshape((m.shape[0], m.shape[1] * m.shape[2], m.shape[3]))

        # 1D: batch_size, 2D: k, 3D: n_h
#        h_m = k_max_pooling(relu(T.dot(m, self.W_m)), k=n_prds, axis=2)
        h_m = T.max(relu(T.dot(m, self.W_m)), axis=1)
        # 1D: batch_size, 2D: n_h
#        h_m = T.mean(T.dot(h_m, self.W_k), axis=1)

        # 1D: batch_size, 2D: k * n_h
        h_m = h_m.reshape((h_m.shape[0], -1)).dimshuffle((0, 'x', 'x', 1))
        # 1D: batch_size, 2D: n_prds, 3D: k * n_h
        h_m = T.repeat(h_m, n_prds, 1)
        # 1D: batch_size, 2D: n_prds, 3D: n_words, 4D: k * n_h
        h_m = T.repeat(h_m, h.shape[0], 2)
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: k * n_h
        h_m = h_m.dimshuffle((2, 0, 1, 3))
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: k * n_h
        h_m = h_m.reshape((h_m.shape[0], h_m.shape[1] * h_m.shape[2], h_m.shape[3]))

        return relu(T.dot(T.concatenate([h, h_m], axis=2), self.W_c))


def k_max_pooling(x, k, axis):
    x = T.sort(x, axis=axis)
    return x[:, -k:]
