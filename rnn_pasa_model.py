from utils import sample_weights, build_shared_zeros, sigmoid, tanh
from optimizers import ada_grad, ada_delta, adam, sgd

import theano
import theano.tensor as T


class Model(object):
    def __init__(self, c, l, n_words, window, opt, lr, init_emb, dim_emb, dim_hidden, n_vocab, L2_reg, unit, activation=tanh):

        self.tr_inputs = [c, l, n_words]
        self.pr_inputs = [c, l, n_words]

        self.c = c  # original c: 1D: batch * n_words, 2D: window; elem=word_id
        self.l = l  # original r: 1D: batch * n_cands; elem=label
        self.n_words = n_words
        self.activation = activation

        batch_size = c.shape[0] / n_words

        self.context = c.reshape((batch_size, n_words, window))
        self.labels = l.reshape((batch_size, n_words))

        self.pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
        else:
            self.emb = theano.shared(init_emb)
        self.E = T.concatenate([self.pad, self.emb], 0)

        self.x = self.E[self.context]

        """
        self.A = theano.shared(sample_weights(max_n_agents, dim_emb))
        self.W_a = theano.shared(sample_weights(dim_emb + dim_hidden, dim_emb))
        self.W_r = theano.shared(sample_weights(dim_emb + dim_hidden, dim_hidden))

        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
            self.params = [self.emb, self.A, self.W_a, self.W_r]
        else:
            self.emb = theano.shared(init_emb)
            self.params = [self.A, self.W_a, self.W_r]

        self.E = T.concatenate([self.pad, self.emb], 0)

        h_c0 = T.zeros(shape=(batch, dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * n_cands, dim_hidden), dtype=theano.config.floatX)

        if unit == 'rnn':
            context_encoder = rnn.Layer(n_i=dim_emb, n_h=dim_hidden)
        else:
            context_encoder = gru.Layer(n_i=dim_emb, n_h=dim_hidden)
        self.params.extend(context_encoder.params)

        x_c = self.E[c]  # 1D: batch, 2D: n_prev_sents, 3D: c_words, 4D: dim_emb
        x_r = self.E[r]  # 1D: batch, 2D: n_cands, 3D: r_words, 4D: dim_emb
        x_a = T.dot(a, self.A).reshape((batch, n_prev_sents, 1, dim_emb))

        context = T.concatenate([x_a, x_c], 2).reshape((batch,  n_prev_sents * (c_words + 1), dim_emb)).dimshuffle(1, 0, 2)
        response = x_r.reshape((batch * n_cands, r_words, dim_emb)).dimshuffle(1, 0, 2)

        # H_c: 1D: c_words, 2D: batch, 3D: dim_hidden
        # H_r: 1D: r_words, 2D: batch * n_cands, 3D: dim_hidden
        H_c, _ = theano.scan(fn=context_encoder.recurrence, sequences=context, outputs_info=h_c0)
        H_r, _ = theano.scan(fn=context_encoder.recurrence, sequences=response, outputs_info=h_r0)

        a_res = T.repeat(self.A[0].dimshuffle('x', 0), batch, 0)
        h_c = H_c[-1]
        A_p = self.A[1:n_agents]  # 1D: batch, 2D: n_agents - 1, 3D: dim_emb
        h_q = H_r[-1].reshape((batch, n_cands, dim_hidden))

        h = T.concatenate([a_res, h_c], 1)  # 1D: batch, 2D: dim_emb + dim_hidden
        score_a = T.dot(T.dot(h, self.W_a), A_p.T)
        score_r = T.batched_dot(T.dot(h, self.W_r), h_q.dimshuffle(0, 2, 1))
        self.p_a = sigmoid(score_a)
        self.p_r = sigmoid(score_r)
        self.p_y_a = self.p_a[T.arange(batch), y_a]
        self.p_n_a = self.p_a[T.arange(batch), n_a]
        self.p_y_r = self.p_r[T.arange(batch), y_r]
        self.p_n_r = self.p_r[T.arange(batch), n_r]

        self.nll_a = loss_function(self.p_y_a, self.p_n_a)
        self.nll_r = loss_function(self.p_y_r, self.p_n_r)
        self.nll = self.nll_r + self.nll_a
        self.L2_sqr = regularization(self.params)
        self.cost = self.nll + L2_reg * self.L2_sqr / 2.

        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

        self.y_a_hat = T.argmax(score_a, axis=1)  # 2D: batch
        self.y_r_hat = T.argmax(score_r, axis=1)  # 1D: batch

        self.correct_a = T.eq(self.y_a_hat, y_a)
        self.correct_r = T.eq(self.y_r_hat, y_r)
        """

