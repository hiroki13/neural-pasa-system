import theano
import theano.tensor as T

from abc import ABCMeta, abstractmethod
from nn_utils import sigmoid, tanh, relu, sample_weights, build_shared_zeros, apply_dropout
from rnn import GRU, LSTM


class EmbeddingLayer(object):

    def __init__(self, init_emb, n_vocab, dim_emb, fix=0, pad=1):
        self.E = None
        self.emb = None
        self.params = []
        self.set_emb(init_emb, n_vocab, dim_emb, fix, pad)

    def set_emb(self, init_emb, n_vocab, dim_emb, fix, pad):
        self.emb = self.create_emb(init_emb, n_vocab, dim_emb, pad)

        if fix:
            self.params.append([])
        else:
            self.params.append(self.emb)

        if pad:
            pad_vec = build_shared_zeros((1, dim_emb))
            self.E = T.concatenate([pad_vec, self.emb], 0)
        else:
            self.E = self.emb

    @staticmethod
    def create_emb(init_emb, n_vocab, dim_emb, pad):
        if init_emb is None:
            n_vocab = n_vocab - 1 if pad else n_vocab
            return theano.shared(sample_weights(n_vocab, dim_emb))
        return theano.shared(init_emb)

    def lookup(self, x):
        return self.E[x]


class Layer(object):

    def __init__(self, n_in=32, n_h=32):
        self.W = theano.shared(sample_weights(n_in, n_h))
        self.params = [self.W]

    def dot(self, x):
        return T.dot(x, self.W)


class RNNLayers(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv, unit, depth, n_in, n_h):
        self.argv = argv
        self.unit = unit.lower()
        self.depth = depth
        self.forward = self.set_forward_func(self.unit)
        self.layers = self.set_layers(self.unit, depth, n_in, n_h)

    @abstractmethod
    def set_forward_func(self, unit):
        raise NotImplementedError

    @abstractmethod
    def set_layers(self, unit, depth, n_in, n_h):
        raise NotImplementedError


class CrankRNNLayers(RNNLayers):
    """
    [Zhou+ 2015]
    """

    def set_forward_func(self, unit):
        if self.unit == 'lstm':
            return self.lstm_forward
        return self.gru_forward

    def set_layers(self, unit, depth, n_in, n_h):
        layer = self.select_layer()
        return [layer(n_in=n_in, n_h=n_h) for i in xrange(depth)]

    def select_layer(self):
        if self.unit == 'lstm':
            return LSTM
        return GRU

    def gru_forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        h0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        for layer in self.layers:
            h = layer.forward_all(x, h0)
            if self.argv.res:
                x = (h + x)[::-1]
            else:
                x = h[::-1]
        if (self.depth % 2) == 1:
            x = x[::-1]
        return x

    def lstm_forward(self, x):
        h0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        c0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        for layer in self.layers:
            h, c = layer.forward_all(x, h0, c0)
            if self.argv.res:
                x = (h + x)[::-1]
            else:
                x = h[::-1]
        if (self.depth % 2) == 1:
            x = x[::-1]
        return x


class BiRNNLayers(RNNLayers):
    """
    [Graves+ 2013]
    """

    def set_forward_func(self, unit):
        return self.gru_forward

    def set_layers(self, unit, depth, n_in, n_h):
        layers = []
        layer = GRU
        for i in xrange(depth):
            layers.append(layer(n_in=n_in, n_h=n_in))
            layers.append(layer(n_in=n_in, n_h=n_in))
        return layers

    def gru_forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_emb
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        h0_f = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        h0_b = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        h = x
        # 1D: n_words, 2D: batch, 3D n_h
        for i in xrange(self.depth):
            hf = self.layers[(2*i)].forward_all(h, h0_f)
            hb = self.layers[(2*i)+1].forward_all(h[::-1], h0_b)[::-1]
            if self.argv.res:
                h = hf + hb + h
            else:
                h = hf + hb
        return h


class BiRNNConcatLayers(RNNLayers):

    def set_forward_func(self, unit):
        return self.gru_forward

    def set_layers(self, unit, depth, n_in, n_h):
        layers = []
        layer = GRU
        layers.append(layer(n_in=n_in, n_h=n_in))
        layers.append(layer(n_in=n_in, n_h=n_in))
        layers.append(Layer(n_in=n_in * 2, n_h=n_h))
        return layers

    def gru_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_emb
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        x = x.dimshuffle(1, 0, 2)
        h0_1 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        h0_2 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        # 1D: n_words, 2D: batch, 3D n_h
        h1 = self.layers[0].forward_all(x, h0_1)
        h2 = self.layers[1].forward_all(x[::-1], h0_2)[::-1]
        return self.layers[2].dot(T.concatenate([h1, h2], axis=2))


class GridCrossNetwork(RNNLayers):

    def set_forward_func(self, unit):
        return self.grid_propagate

    def set_layers(self, unit, depth, n_in, n_h):
        if unit == 'gru':
            layer = GRU
        else:
            layer = LSTM
        layers = []
        for i in xrange(depth):
            layers.extend([layer(n_in=n_h, n_h=n_h) for i in xrange(4)])
        return layers

    def grid_propagate(self, h):
        """
        :param h: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        hf0 = T.zeros((h.shape[0], h.shape[1], h.shape[3]), dtype=theano.config.floatX)
        hb0 = T.zeros((h.shape[0], h.shape[1], h.shape[3]), dtype=theano.config.floatX)
        hu0 = T.zeros((h.shape[0], h.shape[2], h.shape[3]), dtype=theano.config.floatX)
        hd0 = T.zeros((h.shape[0], h.shape[2], h.shape[3]), dtype=theano.config.floatX)
        for i in xrange(0, self.depth):
            hf = self.forward_all(self.layers[i*4], h, hf0)
            hb = self.backward_all(self.layers[(i*4)+1], h, hb0)
            hu = self.upward_all(self.layers[(i*4)+2], h, hu0)
            hd = self.downward_all(self.layers[(i*4)+3], h, hd0)

            if self.argv.res and i != self.depth-1:
                h = self.add(hf, hb, hu, hd) + h
            else:
                h = self.add(hf, hb, hu, hd)
        return h

    @staticmethod
    def forward_all(layer, x, h):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :param h: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_prds, 4D: dim_h
        """
        x = x.dimshuffle(2, 0, 1, 3)
        return layer.forward_all(x, h)

    @staticmethod
    def backward_all(layer, x, h):
        x = x.dimshuffle(2, 0, 1, 3)
        return layer.forward_all(x[::-1], h)

    @staticmethod
    def upward_all(layer, x, h):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :param h: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: n_prds, 2D: batch, 3D: n_words, 4D: dim_h
        """
        x = x.dimshuffle(1, 0, 2, 3)
        return layer.forward_all(x, h)

    @staticmethod
    def downward_all(layer, x, h):
        x = x.dimshuffle(1, 0, 2, 3)
        return layer.forward_all(x[::-1], h)

    @staticmethod
    def dot(layer, hf, hb, hu, hd):
        hf = hf.dimshuffle(1, 2, 0, 3)
        hb = hb.dimshuffle(1, 2, 0, 3)[::-1]
        hu = hu.dimshuffle(1, 0, 2, 3)
        hd = hd.dimshuffle(1, 0, 2, 3)[::-1]
        h = T.concatenate([hf, hb, hu, hd], axis=3)
        return layer.dot(h)

    @staticmethod
    def add(hf, hb, hu, hd):
        hf = hf.dimshuffle(1, 2, 0, 3)
        hb = hb.dimshuffle(1, 2, 0, 3)[::-1]
        hu = hu.dimshuffle(1, 0, 2, 3)
        hd = hd.dimshuffle(1, 0, 2, 3)[::-1]
        return hf + hb + hu + hd


class GridObliqueNetwork(RNNLayers):

    def set_forward_func(self, unit):
        return self.grid_propagate

    def set_layers(self, unit, depth, n_in, n_h):
        layer = self.select_layer()
        return [layer(n_h=n_h) for i in xrange(depth)]

    def select_layer(self):
        argv = self.argv
        if argv.gru_in == 'add':
            return ObliqueForwardNetAdd
        return ObliqueForwardNet

    def grid_propagate(self, h):
        """
        :param h: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        h0_c = T.zeros((h.shape[0], h.shape[3]), dtype=theano.config.floatX)
        h0_r = T.zeros((h.shape[2], h.shape[0], h.shape[3]), dtype=theano.config.floatX)
        h = h.dimshuffle(1, 2, 0, 3)
        for i in xrange(0, self.depth):
            h_tmp = self.layers[i].forward_all(h, h0_r, h0_c)

            if self.argv.res:
                h = h_tmp + h
            else:
                h = h_tmp

            h = self.flip(h)

        if (self.depth % 2) == 1:
            h = self.flip(h)
        return h.dimshuffle(2, 0, 1, 3)

    @staticmethod
    def flip(x):
        x = x[::-1]
        x = x.dimshuffle(1, 0, 2, 3)
        x = x[::-1]
        return x.dimshuffle(1, 0, 2, 3)


class ObliqueForwardNet(object):

    def __init__(self, n_h):
        self.unit = GRU(n_in=n_h*2, n_h=n_h)
        self.params = self.unit.params

    def forward_all(self, x, h_prev, h0):
        """
        :param x: 1D: n_prds, 2D: n_words, 3D: batch, dim_h
        :param h_prev: 1D: n_words, 2D: batch, 3D: dim_h
        :param h0: 1D: batch, 2D: dim_h
        :return: 1D: n_prds, 2D: n_words, 3D: batch, 3D: dim_h
        """
        h, _ = theano.scan(fn=self.forward_row, sequences=[x], outputs_info=[h_prev], non_sequences=[h0])
        return h

    def forward_row(self, x, h_prev, h0):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h_prev: 1D: n_words, 2D: batch, 3D: dim_h
        :param h0: 1D: batch, 2D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        return self.forward_column(T.concatenate([x, h_prev], axis=2), h0)

    def forward_column(self, x, h):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        return self.unit.forward_all(x, h)


class ObliqueForwardNetAdd(object):

    def __init__(self, n_h):
        self.unit = GRU(n_in=n_h, n_h=n_h)
        self.params = self.unit.params

    def forward_all(self, x, h_prev, h0):
        h, _ = theano.scan(fn=self.forward_row, sequences=[x], outputs_info=[h_prev], non_sequences=[h0])
        return h

    def forward_row(self, x, h_prev, h0):
        return self.forward_column(x + h_prev, h0)

    def forward_column(self, x, h):
        return self.unit.forward_all(x, h)


class GridAttentionNetwork(RNNLayers):

    def set_forward_func(self, unit):
        return self.grid_propagate

    def set_layers(self, unit, depth, n_in, n_h):
        layer = AttentionalGRU
        return [layer(n_in=n_in, n_h=n_h) for i in xrange(depth)]

    def grid_propagate(self, h):
        """
        :param h: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        h0 = T.zeros((h.shape[0], h.shape[1], h.shape[3]), dtype=theano.config.floatX)
        h = h.dimshuffle(2, 0, 1, 3)
        for i in xrange(self.depth):
            h_tmp = self.layers[i].forward_all(h, h0)

            if self.argv.res:
                h = h_tmp + h
            else:
                h = h_tmp

            h = h[::-1]

        if (self.depth % 2) == 1:
            h = h[::-1]
        return h.dimshuffle(1, 2, 0, 3)


class AttentionalGRU(object):

    def __init__(self, n_in=32, n_h=32, activation=tanh):
        self.activation = activation

        self.W_xr = theano.shared(sample_weights(n_in, n_h))
        self.W_hr = theano.shared(sample_weights(n_h, n_h))

        self.W_xz = theano.shared(sample_weights(n_in, n_h))
        self.W_hz = theano.shared(sample_weights(n_h, n_h))

        self.W_xh = theano.shared(sample_weights(n_in, n_h))
        self.W_hh = theano.shared(sample_weights(n_h, n_h))

        self.attention = Attention(n_h)

        self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh] + self.attention.params

    def forward(self, xr_t, xz_t, xh_t, h_tm1):
        # 1D: batch, 2D: n_prds, 3D: dim_h
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        self.attention.forward(h_t)
        return h_t

    def forward_all(self, x, h0):
        xr = T.dot(x, self.W_xr)
        xz = T.dot(x, self.W_xz)
        xh = T.dot(x, self.W_xh)
        h, _ = theano.scan(fn=self.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        return h


class Attention(object):

    def __init__(self, n_h):
        self.W1_h = theano.shared(sample_weights(n_h, n_h))
        self.w    = theano.shared(sample_weights(n_h, ))
        self.W2_r = theano.shared(sample_weights(n_h, n_h))
        self.W2_h = theano.shared(sample_weights(n_h, n_h))
        self.params = [self.W1_h, self.w, self.W2_r, self.W2_h]

    def forward(self, h_t):
        """
        :param h_t: 1D: batch_size, 2D n_prds, 3D: n_h
        :return: 1D: batch_size, 2D: n_prds, 3D: n_h
        """

        # 1D: batch_size, 2D: n_prds, 3D: n_h
        M = T.tanh(T.dot(h_t, self.W1_h))

        # 1D: batch_size, 2D: n_prds, 3D: 1
        alpha = T.nnet.softmax(T.dot(M, self.w))
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch_size, 2D: n_h
        r = T.sum(h_t * alpha, axis=1)

#        return T.tanh(T.dot(r, self.W2_r).dimshuffle(0, 'x', 1) + T.dot(h_t, self.W2_h))
        return T.tanh(T.dot(r, self.W2_r)).dimshuffle(0, 'x', 1) + T.dot(h_t, self.W2_h)

