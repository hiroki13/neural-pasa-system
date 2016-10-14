from nn.nn_utils import sample_weights, build_shared_zeros, L2_sqr, sigmoid
from nn.optimizers import ada_grad, ada_delta, adam, sgd
from nn import cnn

import theano
import theano.tensor as T


class Model(object):
    def __init__(self, x, n_cands, window, opt, lr, init_emb, dim_emb, dim_hidden, dim_out,
                 n_vocab, L2_reg, unit, n_layers=2):
        self.tr_inputs = [x, n_cands]
        self.pr_inputs = [x, n_cands]

        self.x = x  # original x: 1D: batch * n_words, 2D: window * 2; elem=word_id

        batch_size = x.shape[0] / n_cands

        self.layers = cnn.layers
        self.W_out = theano.shared(sample_weights(dim_hidden))

        self.pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
        else:
            self.emb = theano.shared(init_emb)
        self.E = T.concatenate([self.pad, self.emb], 0)
        self.params = [self.emb, self.W_out]

        e = self.E[x]
        e_reshape = e.reshape((x.shape[0], -1))

        h, params = self.layers(x=e_reshape, window=window, dim_emb=dim_emb, dim_hidden=dim_hidden, n_layers=n_layers)
        self.params.extend(params)

        self.p_y = T.dot(h, self.W_out).reshape((batch_size, n_cands))
        self.y_pred = T.argmax(self.p_y, 1)

        pos_scores = self.p_y[:, 0]
        neg_scores = T.max(self.p_y[:, 1:], 1)
        diff = neg_scores - pos_scores + 1.0
        loss = T.mean((diff > 0) * diff)

        self.nll = loss
        self.cost = self.nll + L2_reg * L2_sqr(self.params) / 2.
        self.corrects = T.eq(self.y_pred, 0)

        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

