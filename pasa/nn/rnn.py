import theano
import theano.tensor as T

from utils import sigmoid, tanh, relu, sample_weights, apply_dropout
from seq_labeling import CRFLayer
from attention import AttentionLayer
import cnn


def set_layers(x, batch, n_fin, n_h, dropout, n_layers=1):
    """
    :param x: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
    :return: 1D: n_words, 2D: batch, 3D: dim_h
    """
    for i in xrange(n_layers):
        # h0: 1D: batch, 2D: n_h
        if i == 0:
            layer = GRU(n_i=n_fin, n_h=n_h)
            x = layer.dot(x.dimshuffle(1, 0, 2))
            h0 = T.zeros((batch, n_h), dtype=theano.config.floatX)
        else:
            layer = GRU(n_i=n_h*2, n_h=n_h)
            x = layer.dot(T.concatenate([x, h], 2))[::-1]
            h0 = x[0]

        # 1D: n_words, 2D: batch, 3D n_h
        h = layer.forward_all(x, h0)
    return h, x


class ConnectedLayer(object):

    def __init__(self, n_i=32, n_h=32, activation=tanh):
        self.activation = activation
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.params = [self.W]

    def dot(self, x):
        return T.dot(x, self.W)


class GRU(object):

    def __init__(self, n_i=32, n_h=32, activation=tanh):
        self.activation = activation

#        self.W = theano.shared(sample_weights(n_i, n_h))

        self.W_xr = theano.shared(sample_weights(n_h, n_h))
        self.W_hr = theano.shared(sample_weights(n_h, n_h))

        self.W_xz = theano.shared(sample_weights(n_h, n_h))
        self.W_hz = theano.shared(sample_weights(n_h, n_h))

        self.W_xh = theano.shared(sample_weights(n_h, n_h))
        self.W_hh = theano.shared(sample_weights(n_h, n_h))

#        self.params = [self.W, self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]
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

#    def dot(self, x):
#        return relu(T.dot(x, self.W))


def layers_mp(x, batch, n_prds, n_fin, n_h, n_y, dropout, attention, mp_cnn=0, n_layers=1):
    params = []

    ##########################################
    # Bidirectional recurrent stacked layers #
    ##########################################
    for i in xrange(n_layers):
        if i == 0:
            layer = GRU(n_i=n_fin, n_h=n_h)
            layer_input = relu(T.dot(x.dimshuffle(1, 0, 2), layer.W))
            h0 = layer.h0 * T.ones((batch, n_h))  # h0: 1D: batch_size, 2D: n_h
        else:
            layer = GRU(n_i=n_h*2, n_h=n_h)
            # h: 1D: n_words, 2D: batch_size, 3D n_h
            layer_input = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))[::-1]
            h0 = layer_input[0]

        xr = T.dot(layer_input, layer.W_xr)
        xz = T.dot(layer_input, layer.W_xz)
        xh = T.dot(layer_input, layer.W_xh)

        h, _ = theano.scan(fn=layer.forward, sequences=[xr, xz, xh], outputs_info=[h0])
        params.extend(layer.params)

        if dropout is not None:
            h = apply_dropout(h, dropout)

        ############################
        # Multiple seq convolution #
        ############################
        if mp_cnn:
            layer = cnn.Layer(n_h=n_h, pooling=T.max, w_skip=1)
            h = layer.convolution(h=h, n_prds=n_prds)
            params.extend(layer.params)

    ##########################################
    # Taking into account multiple sequences #
    ##########################################
    if attention:
        layer = AttentionLayer(n_h=n_h)
        layer_action = layer.multi_prd_attention
    elif mp_cnn:
        layer = cnn.Layer(n_h=n_h, pooling=T.max)
        layer_action = layer.convolution

    if attention or mp_cnn:
        params.extend(layer.params)
        layer_input = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))
        h = layer_action(h=layer_input, n_prds=n_prds)

    if dropout is not None:
        h = apply_dropout(h, dropout)

    #############
    # CRF layer #
    #############
    layer = CRFLayer(n_i=n_h * 2, n_labels=n_y)
    params.extend(layer.params)
    h = relu(T.dot(T.concatenate([layer_input, h], 2), layer.W))

    if n_layers % 2 == 0:
        emit = h[::-1]
    else:
        emit = h

    return params, layer, emit
