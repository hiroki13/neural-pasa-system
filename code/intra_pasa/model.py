from nn.utils import L2_sqr, relu
from nn.optimizers import ada_grad, ada_delta, adam, sgd
from nn import gru, lstm
from nn.crf import CRFLayer, y_prob, vitabi
from nn.embedding import EmbeddingLayer

import numpy as np
import theano
import theano.tensor as T


class Model(object):

    def __init__(self, x, y, n_words, n_vocab, init_emb, n_in, n_h, n_y,
                 window, unit, opt, lr, dropout, L2_reg, n_layers):

        self.inputs = [x, y, n_words]
        self.dropout = theano.shared(np.float32(dropout).astype(theano.config.floatX))

        params = []
        batch_size = x.shape[0] / n_words
        n_fin = n_in * window
        y_reshaped = self.y_reshaped = y.reshape((batch_size, n_words))

        ###################
        # Embedding layer #
        ###################
        layer = EmbeddingLayer(n_vocab, n_in, init_emb)
        params += layer.params

        x_in = layer.lookup(x)
        x_in = x_in.reshape((batch_size, n_words, n_fin))

        #######################
        # Intermediate layers #
        #######################
        if unit == 'lstm':
            layers = lstm.layers
        else:
            layers = gru.set_layers

        param, h, x_in = layers(x=x_in, batch=batch_size, n_fin=n_fin, n_h=n_h, dropout=self.dropout, n_layers=n_layers)
        params += param

        ################
        # Output layer #
        ################
        layer = CRFLayer(n_i=n_h * 2, n_h=n_y)
        params += layer.params

        h = relu(T.dot(T.concatenate([x_in, h], 2), layer.W))

        if n_layers % 2 == 0:
            h = h[::-1]

        p_y = y_prob(layer, h, y_reshaped.dimshuffle(1, 0), batch_size)

        ###########
        # Predict #
        ###########
        self.y_pred = vitabi(layer, h, batch_size)

        #############
        # Objective #
        #############
        self.nll = - T.mean(p_y)
        cost = self.nll + L2_reg * L2_sqr(params) / 2.

        #############
        # Optimizer #
        #############
        if opt == 'adagrad':
            self.update = ada_grad(cost=cost, params=params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=cost, params=params)
        elif opt == 'adam':
            self.update = adam(cost=cost, params=params)
        else:
            self.update = sgd(cost=cost, params=params, lr=lr)

