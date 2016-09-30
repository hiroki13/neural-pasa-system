import numpy as np
import theano
import theano.tensor as T

from ..utils.io_utils import say
from ..nn.rnn import RNNLayers
from ..nn.utils import L2_sqr
from ..nn.optimizers import ada_grad, ada_delta, adam, sgd
from ..nn.seq_labeling import Layer, MEMMLayer, CRFLayer
from ..nn.embedding import EmbeddingLayer


class Model(object):

    def __init__(self, argv, emb, vocab_word, vocab_label):

        self.argv = argv
        self.emb = emb
        self.n_vocab = vocab_word.size()
        self.n_labels = vocab_label.size()
        self.dropout = None

        ###################
        # Input variables #
        ###################
        self.inputs = None

        ####################
        # Output variables #
        ####################
        self.p_y = None
        self.y_gold = None
        self.y_pred = None
        self.nll = None
        self.cost = None

        ##############
        # Parameters #
        ##############
        self.emb_layer = None
        self.hidden_layers = None
        self.output_layer = None
        self.layers = []
        self.params = []
        self.update = None

    def compile(self, x, y, n_words, n_prds=None):
        argv = self.argv
        init_emb = self.emb

        ###################
        # Input variables #
        ###################
        if n_prds:
            self.inputs = [x, y, n_words, n_prds]
        else:
            self.inputs = [x, y, n_words]

        ##############
        # Dimensions #
        ##############
        batch_size = x.shape[0] / n_words
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_in = dim_emb * (5 + argv.window + 1)
        dim_h = argv.dim_hidden
        dim_out = self.n_labels
        n_vocab = self.n_vocab

        ###################
        # Hyperparameters #
        ###################
        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        lr = argv.lr
        reg = argv.reg
        opt = argv.opt
        unit = argv.unit
        n_layers = argv.layers

        ##############
        # Parameters #
        ##############
        self.set_layers(unit, n_vocab, init_emb, dim_emb, dim_in, dim_h, dim_out, n_layers)
        self.set_params()

        ############
        # Networks #
        ############
        x = self.emb_layer_forward(x, batch_size, n_words)
        h = self.hidden_layer_forward(x)
        h = self.output_layer_forward(h)

        ###########
        # Outputs #
        ###########
        self.y_gold = y.reshape((batch_size, n_words))
        self.y_pred = self.output_layer.decode(h)
        self.p_y = self.output_layer.get_y_prob(h, self.y_gold.dimshuffle((1, 0)))

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(self.p_y, reg)
        self.update = self.optimize(opt, self.cost, lr)

    def set_layers(self, unit, n_vocab, init_emb, n_emb, n_in, n_h, n_y, n_layers):
        self.emb_layer = EmbeddingLayer(n_vocab=n_vocab, dim_emb=n_emb, init_emb=init_emb)
        self.hidden_layers = RNNLayers(unit=unit, depth=n_layers, n_in=n_in, n_h=n_h)

        if self.argv.output_layer == 0:
            self.output_layer = Layer(n_i=n_h, n_labels=n_y)
        elif self.argv.output_layer == 1:
            self.output_layer = MEMMLayer(n_i=n_h, n_labels=n_y)
        else:
            self.output_layer = CRFLayer(n_i=n_h, n_labels=n_y)

        self.layers.append(self.emb_layer)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)

        say('No. of rnn layers: %d\n' % (len(self.layers)-3))

    def set_params(self):
        for l in self.layers:
            self.params += l.params
        say("No. of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def emb_layer_forward(self, x, batch, n_words):
        """
        :param x: 1D: batch * n_words, 2D: 5 + window + 1; elem=word_id
        :return: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        """
        return self.emb_layer.forward(x).reshape((batch, n_words, -1))

    def hidden_layer_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        return self.hidden_layers.forward(x)

    def output_layer_forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        h = self.layers[-1].forward(x)
        if (len(self.layers) - 3) % 2 == 0:
            h = h[::-1]
        return h

    def objective_f(self, p_y, reg):
        nll = - T.mean(p_y)
        cost = nll + reg * L2_sqr(self.params) / 2.
        return nll, cost

    def optimize(self, opt, cost, lr):
        params = self.params
        if opt == 'adagrad':
            return ada_grad(cost=cost, params=params, lr=lr)
        elif opt == 'ada_delta':
            return ada_delta(cost=cost, params=params)
        elif opt == 'adam':
            return adam(cost=cost, params=params)
        return sgd(cost=cost, params=params, lr=lr)
