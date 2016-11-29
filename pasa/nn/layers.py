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

    def __init__(self, n_in=32, n_h=32, activation='relu'):
        self.activation = self.select_activation_func(activation)
        self.W = theano.shared(sample_weights(n_in, n_h))
        self.params = [self.W]

    @staticmethod
    def select_activation_func(activation):
        if activation == 'relu':
            return relu
        elif activation == 'sigmoid':
            return sigmoid
        return tanh

    def dot(self, x):
        return self.activation(T.dot(x, self.W))


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
            x = (h + x)[::-1]
        if (self.depth % 2) == 1:
            x = x[::-1]
        return x

    def lstm_forward(self, x):
        h0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        c0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
        for layer in self.layers:
            h, c = layer.forward_all(x, h0, c0)
            x = (h + x)[::-1]
        if (self.depth % 2) == 1:
            x = x[::-1]
        return x


class BiRNNLayers(RNNLayers):

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
            layers.append(Layer(n_in=n_h * 4, n_h=n_h))
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
            hf = self.forward_all(self.layers[i * 5], h, hf0)
            hb = self.backward_all(self.layers[(i * 5) + 1], h, hb0)
            hu = self.upward_all(self.layers[(i * 5) + 2], h, hu0)
            hd = self.downward_all(self.layers[(i * 5) + 3], h, hd0)
            h = h + self.dot(self.layers[(i * 5) + 4], hf, hb, hu, hd)
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


class GridObliqueNetwork(RNNLayers):

    def set_forward_func(self, unit):
        return self.grid_propagate

    def set_layers(self, unit, depth, n_in, n_h):
        layer = self.select_layer()
        return [layer(n_h=n_h) for i in xrange(depth)]

    def select_layer(self):
        argv = self.argv
        if argv.gru_in == 'add':
            if argv.gru_connect:
                return ObliqueForwardNetAddConnect
            else:
                return ObliqueForwardNetAdd
        else:
            if argv.gru_connect:
                return ObliqueForwardNetConnect
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
            h = h + h_tmp
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


class ObliqueForwardNetConnect(object):

    def __init__(self, n_h):
        self.unit = GRU(n_in=n_h*2, n_h=n_h)
        self.params = self.unit.params

    def forward_all(self, x, h_prev, h0):
        [h, _], _ = theano.scan(fn=self.forward_row, sequences=[x], outputs_info=[h_prev, h0])
        return h

    def forward_row(self, x, h_prev, h0):
        h = self.forward_column(T.concatenate([x, h_prev], axis=2), h0)
        return h, h[-1]

    def forward_column(self, x, h):
        return self.unit.forward_all(x, h)


class ObliqueForwardNetAddConnect(object):

    def __init__(self, n_h):
        self.unit = GRU(n_in=n_h, n_h=n_h)
        self.params = self.unit.params

    def forward_all(self, x, h_prev, h0):
        [h, _], _ = theano.scan(fn=self.forward_row, sequences=[x], outputs_info=[h_prev, h0])
        return h

    def forward_row(self, x, h_prev, h0):
        h = self.forward_column(x + h_prev, h0)
        return h, h[-1]

    def forward_column(self, x, h):
        return self.unit.forward_all(x, h)

