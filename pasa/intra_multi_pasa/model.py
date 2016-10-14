from nn.nn_utils import sample_weights, build_shared_zeros, L2_sqr
from nn.optimizers import ada_grad, ada_delta, adam, sgd
from nn import rnn, lstm
from nn.seq_labeling import y_prob, vitabi

import numpy as np
import theano
import theano.tensor as T


class Model(object):
    def __init__(self, x, y, n_words, n_prds, window, opt, lr, init_emb, dim_emb, dim_hidden, dim_out,
                 n_vocab, L2_reg, unit, dropout, attention, mp_cnn, n_layers=2):
        self.tr_inputs = [x, y, n_words, n_prds]
        self.pr_inputs = [x, y, n_words, n_prds]

        self.x = x  # original x: 1D: batch * n_words, 2D: window + 1; elem=word_id
        self.y = y  # original y: 1D: batch * n_cands; elem=label

        batch_size = x.shape[0] / n_words
        n_fin = dim_emb * window
        if dropout is not None:
            dropout_prob = theano.shared(np.float32(dropout).astype(theano.config.floatX))
        else:
            dropout_prob = None
        self.y_reshaped = y.reshape((batch_size, n_words))

        if unit == 'lstm':
            self.layers = lstm.layers
        else:
            self.layers = rnn.layers_mp

        self.pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
        else:
            self.emb = theano.shared(init_emb)
        self.E = T.concatenate([self.pad, self.emb], 0)
        self.params = [self.emb]

        e = self.E[x]
        e_reshaped = e.reshape((batch_size, n_words, n_fin))

        params, o_layer, emit = self.layers(x=e_reshaped, batch=batch_size, n_prds=n_prds, n_fin=n_fin, n_h=dim_hidden,
                                            n_y=dim_out, dropout=dropout_prob, n_layers=n_layers, attention=attention,
                                            mp_cnn=mp_cnn)
        self.params.extend(params)

        self.p_y = y_prob(o_layer, emit, self.y_reshaped.dimshuffle(1, 0), batch_size)
        self.y_pred = vitabi(o_layer, emit, batch_size)

        self.nll = - T.mean(self.p_y)
        self.cost = self.nll + L2_reg * L2_sqr(self.params) / 2.
        self.corrects = T.eq(self.y_pred, self.y_reshaped)

        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

