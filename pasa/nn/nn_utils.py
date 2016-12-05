import random

import numpy as np
import theano
import theano.tensor as T


default_rng = np.random.RandomState(random.randint(0, 9999))
default_srng = T.shared_randomstreams.RandomStreams(default_rng.randint(9999))


def relu(x):
    return T.nnet.relu(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def softmax_3d(x, axis=0, keepdim=False, eps=1e-8):
    x_e = T.exp(x)
    return x_e / T.sum(x_e, axis=axis, keepdims=keepdim) + eps


def sample_weights(size_x, size_y=0, sig=False):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    if sig:
        W *= 4.0
    return W


def build_shared_zeros(shape):
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        borrow=True
    )


def logsumexp(x, axis):
    """
    :param x: 1D: batch, 2D: n_y, 3D: n_y
    :return: 1D: batch, 2D: 1, 3D: n_y
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def logsumexp_3d(x):
    """
    :param x: 1D: batch, 2D: n_y, 3D: n_y
    :return: 1D: batch, 2D: n_y
    """
    x_max = T.max(x, axis=1, keepdims=True)
    alpha_t = T.log(T.sum(T.exp(x - x_max), axis=1, keepdims=True)) + x_max
    return alpha_t.reshape((alpha_t.shape[0], alpha_t.shape[2]))


def L2_sqr(params):
    return reduce(lambda a, b: a + T.sum(b ** 2), params, 0.)


def normalize_2d(x, eps=1e-8):
    # x is batch*d
    # l2 is batch*1
    l2 = x.norm(2, axis=1).dimshuffle((0, 'x'))
    return x / (l2 + eps)


def normalize_3d(x, eps=1e-8):
    # x is len*batch*d
    # l2 is len*batch*1
    l2 = x.norm(2, axis=2).dimshuffle((0, 1, 'x'))
    return x / (l2 + eps)


def binary_cross_entropy(y, y_p):
    return - (T.sum(y * T.log(y_p)) + T.sum((1 - y) * T.log(1. - y_p)))


def hinge_loss(pos_scores, neg_scores):
    """
    :param pos_scores: 1D: batch, 2D: n_labels
    :param neg_scores: 1D: batch, 2D: n_labels
    :return: avg hinge_loss: float
    """
    loss = 1.0 + neg_scores - pos_scores
    return T.mean((loss > 0) * loss)


class Dropout(object):
    def __init__(self, dropout_prob, srng=None, v2=False):
        """
        :param dropout_prob: theano shared variable that stores the dropout probability
        :param srng: theano random stream or None (default rng will be used)
        :param v2: which dropout version to use
        """
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1 - self.dropout_prob) if not self.v2 else (1 - self.dropout_prob) ** 0.5
        mask = self.srng.binomial(
                n=1,
                p=1-self.dropout_prob,
                size=x.shape,
                dtype=theano.config.floatX
            )
        return x * mask / d


def apply_dropout(x, dropout_prob, v2=False):
    return Dropout(dropout_prob, v2=v2).forward(x)

