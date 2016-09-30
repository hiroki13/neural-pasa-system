import theano
import theano.tensor as T

from utils import sigmoid, tanh, relu, sample_weights, apply_dropout


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
