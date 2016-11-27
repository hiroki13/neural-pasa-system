import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, sample_weights


class GRU(object):

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

    def __init__(self, n_in, n_h, activation=tanh):
        self.activation = activation

        self.W_xi = theano.shared(sample_weights(n_in, n_h))
        self.W_hi = theano.shared(sample_weights(n_h, n_h))
        self.W_ci = theano.shared(sample_weights(n_h))

        self.W_xf = theano.shared(sample_weights(n_in, n_h))
        self.W_hf = theano.shared(sample_weights(n_h, n_h))
        self.W_cf = theano.shared(sample_weights(n_h))

        self.W_xc = theano.shared(sample_weights(n_in, n_h))
        self.W_hc = theano.shared(sample_weights(n_h, n_h))

        self.W_xo = theano.shared(sample_weights(n_in, n_h))
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
