from nn.utils import sample_weights, build_shared_zeros, L2_sqr
from nn.optimizers import ada_grad, ada_delta, adam, sgd
from nn import gru, lstm
from nn.crf import y_prob, vitabi

import numpy as np
import theano
import theano.tensor as T


class Model(object):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        self.argv = argv

        ###################
        # Input variables #
        ###################
        # x: 1D: batch * n_words, 2D: window + 1; elem=word_id
        # y: 1D: batch * n_cands; elem=label
        self.x = T.imatrix('x')
        self.y = T.ivector('y')
        self.n_words = T.iscalar('n_words')
        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.inputs = [self.x, self.y, self.n_words]

        ###################
        # Hyperparameters #
        ###################
#        self.window = argv.window * 2 + 1
        self.window = 5 + argv.window + 1
        self.opt = argv.opt
        self.lr = argv.lr
        self.init_emb = emb
        self.dim_emb = argv.dim_emb if emb is None else len(emb[0])
        self.dim_hidden = argv.dim_hidden
        self.dim_out = vocab_label.size()
        self.n_vocab = vocab_word.size()
        self.L2_reg = argv.reg
        self.unit = argv.unit
        self.attention = argv.attention
        self.n_layers = argv.layer

        #############
        # Functions #
        #############
        self.train = None
        self.predict = None

        ###########
        # Outputs #
        ###########
        self.params = []
        self.y_reshaped = None
        self.E = None
        self.nll = None
        self.y_pred = None
        self.update = None

    def compile(self):
        batch_size = self.x.shape[0] / self.n_words
        n_fin = self.dim_emb * self.window
        self.y_reshaped = self.y.reshape((batch_size, self.n_words))

        ##############
        # Set layers #
        ##############
        if self.unit == 'lstm':
            layers = lstm.layers
        else:
            layers = gru.layers

        ###########################
        # Set the embedding layer #
        ###########################
        pad = build_shared_zeros((1, self.dim_emb))
        if self.init_emb is None:
            emb = theano.shared(sample_weights(self.n_vocab - 1, self.dim_emb))
        else:
            emb = theano.shared(self.init_emb)
        self.E = T.concatenate([pad, emb], 0)
        self.params.append(emb)

        ###############
        # Input layer #
        ###############
        x_in = self.E[self.x]
        x_in = x_in.reshape((batch_size, self.n_words, n_fin))

        #######################
        # Intermediate layers #
        #######################
        params, o_layer, emit = layers(x=x_in, batch=batch_size, n_fin=n_fin, n_h=self.dim_hidden, n_y=self.dim_out,
                                       dropout=self.dropout, attention=self.attention, n_layers=self.n_layers)
        self.params.extend(params)

        ################
        # Output layer #
        ################
        p_y = y_prob(o_layer, emit, self.y_reshaped.dimshuffle(1, 0), batch_size)
        self.y_pred = vitabi(o_layer, emit, batch_size)

        #############
        # Objective #
        #############
        self.nll = - T.mean(p_y)
        cost = self.nll + self.L2_reg * L2_sqr(self.params) / 2.

        #############
        # Optimizer #
        #############
        if self.opt == 'adagrad':
            self.update = ada_grad(cost=cost, params=self.params, lr=self.lr)
        elif self.opt == 'ada_delta':
            self.update = ada_delta(cost=cost, params=self.params)
        elif self.opt == 'adam':
            self.update = adam(cost=cost, params=self.params)
        else:
            self.update = sgd(cost=cost, params=self.params, lr=self.lr)

    def set_train_f(self, samples):
        index = T.iscalar('index')
        bos = T.iscalar('bos')
        eos = T.iscalar('eos')

        self.train = theano.function(inputs=[index, bos, eos],
                                     outputs=[self.y_pred, self.y_reshaped, self.nll],
                                     updates=self.update,
                                     givens={
                                         self.inputs[0]: samples[0][bos: eos],
                                         self.inputs[1]: samples[1][bos: eos],
                                         self.inputs[2]: samples[2][index],
                                     }
                                     )

    def set_train_online_f(self):
        self.train = theano.function(inputs=self.inputs,
                                     outputs=[self.y_pred, self.y_reshaped, self.nll],
                                     updates=self.update,
                                     )

    def set_predict_f(self):
        self.predict = theano.function(inputs=self.inputs,
                                       outputs=[self.y_pred, self.y_reshaped],
                                       )

