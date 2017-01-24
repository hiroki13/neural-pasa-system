import theano
import theano.tensor as T

from abc import ABCMeta, abstractmethod
from nn_utils import sample_weights, build_shared_zeros
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


class SoftmaxLayer(object):

    def __init__(self, n_i, n_labels):
        self.W = theano.shared(sample_weights(n_i, n_labels))
        self.params = [self.W]

    def forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels; log probability of a label
        """
        # 1D: n_words, 2D: batch, 3D: n_labels
        h = T.dot(x, self.W)
        # 1D: n_words * batch, 2D: n_labels
        h_reshaped = h.reshape((h.shape[0] * h.shape[1], h.shape[2]))
        return T.log(T.nnet.softmax(h_reshaped).reshape((h.shape[0], h.shape[1], -1)))

    def get_y_prob(self, h, y):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels
        :param y: 1D: n_words, 2D: batch
        :return: 1D: batch; log probability of the correct sequence
        """
        emit_scores = self._get_emit_score(h, y)
        return T.sum(emit_scores, axis=0)

    @staticmethod
    def _get_emit_score(h, y):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
        :param y: 1D: n_words, 2D: batch; label id
        :return: 1D: n_words, 2D: batch; specified label score
        """
        # 1D: n_words * batch, 2D: n_labels
        h = h.reshape((h.shape[0] * h.shape[1], -1))
        return h[T.arange(h.shape[0]), y.ravel()].reshape(y.shape)

    @staticmethod
    def decode(h):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels; log probability of a label
        :return: 1D: batch, 2D: n_words; the highest scoring sequence (label id)
        """
        return T.argmax(h, axis=2).dimshuffle(1, 0)


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


class BiRNNLayers(RNNLayers):

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


class ConcatedBiRNNLayers(RNNLayers):

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


class StackedBiRNNLayers(RNNLayers):

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


class GridNetwork(RNNLayers):

    def set_forward_func(self, unit):
        return self.grid_propagate

    def set_layers(self, unit, depth, n_in, n_h):
        layer = ObliqueForwardNet
        return [layer(n_h=n_h) for i in xrange(depth)]

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

