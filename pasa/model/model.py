import numpy as np
import theano
import theano.tensor as T

from ..utils.io_utils import say
from ..nn.rnn import RNNLayers
from ..nn.nn_utils import L2_sqr, hinge_loss
from ..nn.optimizers import ada_grad, ada_delta, adam, sgd
from ..nn.seq_labeling import Layer, MEMMLayer, CRFLayer, RankingLayer
from ..nn.embedding import EmbeddingLayer


class Model(object):

    def __init__(self, argv, emb, n_vocab, n_labels):

        self.argv = argv
        self.emb = emb
        self.n_vocab = n_vocab
        self.n_labels = n_labels
        self.dropout = None

        ###################
        # Input variables #
        ###################
        self.inputs = None

        ####################
        # Output variables #
        ####################
        self.y_prob = None
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

    def compile(self, x_w, x_p, y, n_words, n_prds=None):
        argv = self.argv
        batch_size = x_w.shape[0] / n_words

        ###################
        # Input variables #
        ###################
        if n_prds:
            self.inputs = [x_w, x_p, y, n_words, n_prds]
        else:
            self.inputs = [x_w, x_p, y, n_words]

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(x_w, n_words, self.emb)
        self.set_params()

        ############
        # Networks #
        ############
        x = self.emb_layer_forward(x_w, x_p, batch_size, n_words)
        h = self.hidden_layer_forward(x)
        h = self.output_layer_forward(h)

        ###########
        # Outputs #
        ###########
        self.y_gold = y.reshape((batch_size, n_words))
        self.y_pred = self.output_layer.decode(h)
        self.y_prob = h.dimshuffle(1, 0, 2)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(h=h, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self, x_w, n_words, init_emb):
        argv = self.argv
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb * (5 + argv.window) + dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels
        n_vocab = self.n_vocab
        unit = argv.unit
        fix = argv.fix
        n_layers = argv.layers

        self.emb_layer = EmbeddingLayer(n_vocab=n_vocab, dim_emb=dim_emb, init_emb=init_emb, dim_posit=dim_posit, fix=fix)
        self.hidden_layers = RNNLayers(unit=unit, depth=n_layers, n_in=dim_in, n_h=dim_h)

        if self.argv.output_layer == 0:
            self.output_layer = Layer(n_i=dim_h, n_labels=dim_out)
        elif self.argv.output_layer == 1:
            self.output_layer = MEMMLayer(n_i=dim_h, n_labels=dim_out)
        else:
            self.output_layer = CRFLayer(n_i=dim_h, n_labels=dim_out)

        self.layers.append(self.emb_layer)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)
        say('No. of rnn layers: %d\n' % (len(self.layers)-3))

    def set_params(self):
        for l in self.layers:
            self.params += l.params
        say("No. of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def emb_layer_forward(self, x_w, x_p, batch, n_words):
        """
        :param x_w: 1D: batch * n_words, 2D: 5 + window; elem=word_id
        :param x_p: 1D: batch * n_words; elem=posit_id
        :return: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        """
        x_w = self.emb_layer.forward_word(x_w).reshape((batch, n_words, -1))
        x_p = self.emb_layer.forward_posit(x_p).reshape((batch, n_words, -1))
        return T.concatenate([x_w, x_p], axis=2)

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

    def objective_f(self, h, reg):
        p_y = self.output_layer.get_y_prob(h, self.y_gold.dimshuffle((1, 0)))
        nll = - T.mean(p_y)
        cost = nll + reg * L2_sqr(self.params) / 2.
        return nll, cost

    def optimize(self, cost, opt, lr):
        params = self.params
        if opt == 'adagrad':
            return ada_grad(cost=cost, params=params, lr=lr)
        elif opt == 'ada_delta':
            return ada_delta(cost=cost, params=params)
        elif opt == 'adam':
            return adam(cost=cost, params=params)
        return sgd(cost=cost, params=params, lr=lr)


class RankingModel(Model):

    def __init__(self, argv, emb, n_vocab, n_labels):
        super(RankingModel, self).__init__(argv, emb, n_vocab, n_labels)

    def compile(self, x_w, x_p, y, n_words):
        self.inputs = [x_w, x_p, y, n_words]

        argv = self.argv
        batch_size = x_w.shape[0] / n_words

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(x_w, n_words, self.emb)
        self.set_params()

        ############
        # Networks #
        ############
        x = self.emb_layer_forward(x_w, x_p, batch_size, n_words)
        h = self.hidden_layer_forward(x)
        h = self.output_layer_forward(h)

        # 1D: 1, 2D: batch, 3D: dim_h
        NULL = T.zeros(shape=(1, h.shape[1], h.shape[2]))
        h = T.concatenate([h, NULL], 0)

        ###########
        # Outputs #
        ###########
        self.y_gold = y
        self.y_pred = self.output_layer.decode(h)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(h=h, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self, x_w, n_words, init_emb):
        argv = self.argv
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb * (5 + argv.window) + dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels
        n_vocab = self.n_vocab
        unit = argv.unit
        fix = argv.fix
        n_layers = argv.layers

        self.emb_layer = EmbeddingLayer(n_vocab=n_vocab, dim_emb=dim_emb, init_emb=init_emb, dim_posit=dim_posit, fix=fix)
        self.hidden_layers = RNNLayers(unit=unit, depth=n_layers, n_in=dim_in, n_h=dim_h)
        self.output_layer = RankingLayer(n_i=dim_h, n_labels=dim_out)

        self.layers.append(self.emb_layer)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)
        say('No. of rnn layers: %d\n' % (len(self.layers)-3))

    def objective_f(self, h, reg):
        pos_scores = self.output_layer.get_y_scores(h, self.y_gold)
        neg_scores = self.output_layer.get_y_hat_scores(h)
        nll = hinge_loss(pos_scores, neg_scores)
        cost = nll + reg * L2_sqr(self.params) / 2.
        return nll, cost


class RerankingModel(object):

    def __init__(self, argv, emb, n_vocab):

        self.argv = argv
        self.emb = emb
        self.n_vocab = n_vocab
        self.dropout = None

        ###################
        # Input variables #
        ###################
        self.inputs = None

        ####################
        # Output variables #
        ####################
        self.y_prob = None
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

    def compile(self, x_w, x_p, x_l, x_s, y):
        """
        :param x_w: 1D: batch, 2D: n_lists, 3D: n_prds, 4D: n_words, 5D: 5+window; word id
        :param x_p: 1D: batch, 2D: n_lists, 3D: n_prds, 4D: n_words; 0/1
        :param x_l: 1D: batch, 2D: n_lists, 3D: n_prds, 4D: n_words; label id
        :param x_s: 1D: batch, 2D: n_lists, 3D: n_prds, 4D: n_words; score
        :param y: 1D: batch; index of the best f1 list
        :return:
        """
        argv = self.argv
        batch_size = x_w.shape[0]

        self.inputs = [x_w, x_p, x_l, x_s, y]

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(x_w, self.emb)
        self.set_params()

        ############
        # Networks #
        ############
        x = self.emb_layer_forward(x_w, x_p, batch_size)
        h = self.hidden_layer_forward(x)
        h = self.output_layer_forward(h)

        ###########
        # Outputs #
        ###########
        self.y_gold = y
        self.y_pred = self.output_layer.decode(h)
        self.y_prob = h.dimshuffle(1, 0, 2)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(h=h, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)
