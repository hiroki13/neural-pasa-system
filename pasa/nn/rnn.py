import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, relu, sample_weights, apply_dropout


class ConnectedLayer(object):

    def __init__(self, n_i=32, n_h=32, activation=tanh):
        self.activation = activation
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.params = [self.W]

    def dot(self, x):
        return T.dot(x, self.W)


class RNNLayers(object):

    def __init__(self, unit, depth, n_in, n_h):
        self.unit = unit.lower()
        self.depth = depth
        self.n_in = n_in
        self.n_h = n_h
        self.forward = self.set_forward_func(self.unit)
        self.layers = self.set_layers(self.unit, depth, n_in, n_h)

    def set_forward_func(self, unit):
        if unit == 'gru':
            return self.gru_forward
        return self.lstm_forward

    @staticmethod
    def set_layers(unit, depth, n_in, n_h):
        if unit == 'gru':
            layer = GRU
        else:
            layer = LSTM

        layers = []
        for i in xrange(depth):
            if i == 0:
                layers.append(ConnectedLayer(n_i=n_in, n_h=n_h))
            layers.append(layer(n_h=n_h))
        return layers

    def gru_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        for i in xrange(0, self.depth):
            # h0: 1D: batch, 2D: n_h
            if i == 0:
                layer = self.layers[i]
                x = layer.dot(x.dimshuffle(1, 0, 2))
                h0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
            else:
                x = (h + x)[::-1]
                h0 = x[0]
            layer = self.layers[i+1]
            # 1D: n_words, 2D: batch, 3D n_h
            h = layer.forward_all(x, h0)
        return x + h

    def lstm_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        for i in xrange(0, self.depth):
            # h0: 1D: batch, 2D: n_h
            if i == 0:
                layer = self.layers[i]
                x = layer.dot(x.dimshuffle(1, 0, 2))
                h0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
                c0 = T.zeros((x.shape[1], x.shape[2]), dtype=theano.config.floatX)
            else:
                x = (h + x)[::-1]
                h0 = x[0]
                c0 = c[-1]
            layer = self.layers[i+1]
            # 1D: n_words, 2D: batch, 3D n_h
            h, c = layer.forward_all(x, h0, c0)
        return x + h


class BiRNNLayers(object):

    def __init__(self, unit, depth, n_in, n_h):
        self.unit = unit.lower()
        self.depth = depth
        self.n_in = n_in
        self.n_h = n_h
        self.forward = self.set_forward_func(self.unit)
        self.layers = self.set_layers(self.unit, depth, n_in, n_h)

    def set_forward_func(self, unit):
        return self.gru_forward

    @staticmethod
    def set_layers(unit, depth, n_in, n_h):
        layer = GRU
        layers = []
        layers.append(layer(n_h=n_in))
        layers.append(layer(n_h=n_in))
        layers.append(ConnectedLayer(n_i=n_in*2, n_h=n_h))
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

    def __init__(self, unit, depth, n_in, n_h):
        super(GridCrossNetwork, self).__init__(unit, depth, n_in, n_h)

    def set_forward_func(self, unit):
        return self.grid_propagate

    @staticmethod
    def set_layers(unit, depth, n_in, n_h):
        if unit == 'gru':
            layer = GRU
        else:
            layer = LSTM
        layers = []
        for i in xrange(depth):
            layers.extend([layer(n_h=n_h) for i in xrange(4)])
            layers.append(ConnectedLayer(n_i=n_h*4, n_h=n_h))
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
            hf = self.forward_all(self.layers[i*5], h, hf0)
            hb = self.backward_all(self.layers[(i*5)+1], h, hb0)
            hu = self.upward_all(self.layers[(i*5)+2], h, hu0)
            hd = self.downward_all(self.layers[(i*5)+3], h, hd0)
            h = h + self.dot(self.layers[(i*5)+4], hf, hb, hu, hd)
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

    def __init__(self, unit, depth, n_in, n_h):
        super(GridObliqueNetwork, self).__init__(unit, depth, n_in, n_h)

    def set_forward_func(self, unit):
        return self.grid_propagate

    @staticmethod
    def set_layers(unit, depth, n_in, n_h):
        layers = []
        for i in xrange(depth):
            layers.append(ObliqueForwardNet(n_h=n_h))
        return layers

    def grid_propagate(self, h):
        """
        :param h: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        h0 = T.zeros((h.shape[0], h.shape[3]), dtype=theano.config.floatX)
        hp0 = T.zeros((h.shape[2], h.shape[0], h.shape[3]), dtype=theano.config.floatX)
        h = h.dimshuffle(1, 2, 0, 3)
        for i in xrange(0, self.depth):
            h_tmp = self.layers[i].forward_all(h, hp0, h0)
            h = h + h_tmp
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
        self.unit = GridGRU(n_in=n_h*2, n_h=n_h)
        self.params = self.unit.params

    def forward_all(self, h, h_prev, h0):
        """
        :param h: 1D: n_prds, 2D: n_words, 3D: batch, dim_h
        :param h_prev: 1D: n_words, 2D: batch, 3D: dim_h
        :param h0: 1D: batch, 2D: dim_h
        :return: 1D: n_prds, 2D: n_words, 3D: batch, 3D: dim_h
        """
        h, _ = theano.scan(fn=self.forward_row, sequences=[h], outputs_info=[h_prev], non_sequences=[h0])
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


class GRU(object):

    def __init__(self, n_h=32, activation=tanh):
        self.activation = activation

        self.W_xr = theano.shared(sample_weights(n_h, n_h))
        self.W_hr = theano.shared(sample_weights(n_h, n_h))

        self.W_xz = theano.shared(sample_weights(n_h, n_h))
        self.W_hz = theano.shared(sample_weights(n_h, n_h))

        self.W_xh = theano.shared(sample_weights(n_h, n_h))
        self.W_hh = theano.shared(sample_weights(n_h, n_h))

        self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]

    def forward(self, xr_t, xz_t, xh_t, h_tm1):
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t

    def forward_all(self, x, h0):
        xr = T.dot(x, self.W_xr)
        xz = T.dot(x, self.W_xz)
        xh = T.dot(x, self.W_xh)
        h, _ = theano.scan(fn=self.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        return h


class GridGRU(object):

    def __init__(self, n_in=32, n_h=32, activation=tanh):
        self.activation = activation

        self.W_xr = theano.shared(sample_weights(n_in, n_h))
        self.W_hr = theano.shared(sample_weights(n_h, n_h))

        self.W_xz = theano.shared(sample_weights(n_in, n_h))
        self.W_hz = theano.shared(sample_weights(n_h, n_h))

        self.W_xh = theano.shared(sample_weights(n_in, n_h))
        self.W_hh = theano.shared(sample_weights(n_h, n_h))

        self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]

    def forward(self, xr_t, xz_t, xh_t, h_tm1):
        r_t = sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(xh_t + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t

    def forward_all(self, x, h0):
        xr = T.dot(x, self.W_xr)
        xz = T.dot(x, self.W_xz)
        xh = T.dot(x, self.W_xh)
        h, _ = theano.scan(fn=self.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        return h


class LSTM(object):

    def __init__(self, n_h, activation=tanh):
        self.activation = activation

        self.W_xi = theano.shared(sample_weights(n_h, n_h))
        self.W_hi = theano.shared(sample_weights(n_h, n_h))
        self.W_ci = theano.shared(sample_weights(n_h))

        self.W_xf = theano.shared(sample_weights(n_h, n_h))
        self.W_hf = theano.shared(sample_weights(n_h, n_h))
        self.W_cf = theano.shared(sample_weights(n_h))

        self.W_xc = theano.shared(sample_weights(n_h, n_h))
        self.W_hc = theano.shared(sample_weights(n_h, n_h))

        self.W_xo = theano.shared(sample_weights(n_h, n_h))
        self.W_ho = theano.shared(sample_weights(n_h, n_h))
        self.W_co = theano.shared(sample_weights(n_h))

        self.params = [self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
                       self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_co]

    def forward(self, xi_t, xf_t, xc_t, xo_t, h_tm1, c_tm1):
        i_t = sigmoid(xi_t + T.dot(h_tm1, self.W_hi) + c_tm1 * self.W_ci)
        f_t = sigmoid(xf_t + T.dot(h_tm1, self.W_hf) + c_tm1 * self.W_cf)
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(xo_t + T.dot(h_tm1, self.W_ho) + c_t * self.W_co)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def forward_all(self, x, h0, c0):
        xi = T.dot(x, self.W_xi)
        xf = T.dot(x, self.W_xf)
        xc = T.dot(x, self.W_xc)
        xo = T.dot(x, self.W_xo)
        [h, c], _ = theano.scan(fn=self.forward, sequences=[xi, xf, xc, xo], outputs_info=[h0, c0])
        return h, c
