import theano.tensor as T

from abc import ABCMeta, abstractmethod
from ..utils.io_utils import say
from ..nn.layers import Layer, EmbeddingLayer, SoftmaxLayer, StackedBiRNNLayers, GridNetwork
from ..nn.nn_utils import L2_sqr, tanh
from ..nn.optimizers import ada_grad, ada_delta, adam, sgd


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv, emb, n_vocab, n_labels):

        self.argv = argv
        self.emb = emb
        self.n_vocab = n_vocab
        self.n_labels = n_labels

        ###################
        # Input variables #
        ###################
        self.inputs = None
        self.x = None

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
        self.emb_layers = []
        self.hidden_layers = None
        self.output_layer = None
        self.layers = []
        self.params = []
        self.update = None

    @abstractmethod
    def compile(self, variables):
        raise NotImplementedError

    @abstractmethod
    def set_layers(self):
        raise NotImplementedError

    def set_params(self):
        for l in self.layers:
            self.params.extend(l.params)
        say("No. of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def optimize(self, cost, opt, lr):
        params = self.params
        if opt == 'adagrad':
            return ada_grad(cost=cost, params=params, lr=lr)
        elif opt == 'ada_delta':
            return ada_delta(cost=cost, params=params)
        elif opt == 'adam':
            return adam(cost=cost, params=params)
        return sgd(cost=cost, params=params, lr=lr)

    def objective_f(self, o, reg):
        p_y = self.output_layer.get_y_prob(o, self.y_gold.dimshuffle((1, 0)))
        nll = - T.mean(p_y)
        cost = nll + reg * L2_sqr(self.params) / 2.
        return nll, cost


class BaseModel(Model):

    def compile(self, variables):
        argv = self.argv

        x = variables[:-1]
        y = variables[-1]

        ###################
        # Input variables #
        ###################
        self.inputs = x + [y]
        self.x = x

        self.set_layers()
        self.set_params()

        ############
        # Networks #
        ############
        h0 = self.emb_layer_forward(x)
        h = self.hidden_layer_forward(h0)
        o = self.output_layer_forward(h)

        ###########
        # Outputs #
        ###########
        self.y_gold = y
        self.y_pred = self.output_layer.decode(o)
        self.y_prob = o.dimshuffle(1, 0, 2)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(o=o, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self):
        argv = self.argv
        dim_emb = argv.dim_emb if self.emb is None else len(self.emb[0])
        dim_posit = argv.dim_posit

        dim_in = dim_emb * (1 + argv.window)
        if len(self.x) > 1:
            dim_in += dim_posit

        dim_h = argv.dim_hidden
        dim_out = self.n_labels

        self.emb_layers.append(EmbeddingLayer(init_emb=self.emb, n_vocab=self.n_vocab, dim_emb=dim_emb,
                                              fix=argv.fix, pad=1))
        if len(self.x) > 1:
            self.emb_layers.append(EmbeddingLayer(init_emb=None, n_vocab=2, dim_emb=dim_posit, fix=argv.fix, pad=0))
        self.emb_layers.append(Layer(n_in=dim_in, n_h=dim_h))
        self.hidden_layers = StackedBiRNNLayers(argv=argv, unit=argv.unit, depth=argv.layers, n_in=dim_h, n_h=dim_h)
        self.output_layer = SoftmaxLayer(n_i=dim_h, n_labels=dim_out)

        self.layers.extend(self.emb_layers)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)
        say('No. of layers: %d\n' % self.argv.layers)

    def emb_layer_forward(self, x):
        """
        :param x: 1D: n_phi, 2D: batch, 3D: n_words, 4D: dim_phi
        :return: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        """
        x_w = x[0]
        x_in = self.emb_layers[0].lookup(x_w).reshape((x_w.shape[0], x_w.shape[1], -1))
        if len(x) > 1:
            x_p = x[1]
            x_p = self.emb_layers[1].lookup(x_p)
            x_in = T.concatenate([x_in, x_p], axis=2)

        return self.emb_layers[len(x)].dot(x_in).dimshuffle(1, 0, 2)

    def hidden_layer_forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_in (dim_emb * (5 + window + 1))
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        return self.hidden_layers.forward(x)

    def output_layer_forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels
        """
        return self.layers[-1].forward(x)


class GridModel(Model):

    def compile(self, variables):
        argv = self.argv
        # x_w: 1D: batch, 2D: n_prds, 3D: n_words, 4D: 5+window; word id
        # x_p: 1D: batch, 2D: n_prds, 3D: n_words; posit id
        # y: 1D: batch, 2D: n_prds, 3D: n_words; elem=label id

        x = variables[:-1]
        y = variables[-1]
        self.inputs = x + [y]
        self.x = x

        self.set_layers()
        self.set_params()

        h0 = self.emb_layer_forward(x)
        h = self.hidden_layer_forward(h0)
        o = self.output_layer_forward(h)

        self.y_pred = self.output_layer.decode(o)
        self.y_gold = y.reshape(self.y_pred.shape)
        self.y_prob = o.dimshuffle(1, 0, 2)

        self.nll, self.cost = self.objective_f(o=o, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self):
        argv = self.argv
        dim_emb = argv.dim_emb if self.emb is None else len(self.emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb * (1 + argv.window)
        if len(self.x) > 1:
            dim_in += dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels

        self.emb_layers.append(EmbeddingLayer(init_emb=self.emb, n_vocab=self.n_vocab, dim_emb=dim_emb,
                                              fix=argv.fix, pad=1))
        if len(self.x) > 1:
            self.emb_layers.append(EmbeddingLayer(init_emb=None, n_vocab=2, dim_emb=dim_posit, fix=argv.fix, pad=0))
        self.emb_layers.append(Layer(n_in=dim_in, n_h=dim_h))
        self.hidden_layers = GridNetwork(argv=argv, unit=argv.unit, depth=argv.layers, n_in=dim_h, n_h=dim_h)
        self.output_layer = SoftmaxLayer(n_i=dim_h, n_labels=dim_out)

        self.layers.extend(self.emb_layers)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)
        say('No. of rnn layers: %d\n' % self.argv.layers)

    def emb_layer_forward(self, x):
        """
        :param x: 1D: n_phi, 2D: batch, 3D: n_prds, 4D: n_words, 5D: dim_phi
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        x_w = x[0]
        x_in = self.emb_layers[0].lookup(x_w).reshape((x_w.shape[0], x_w.shape[1], x_w.shape[2], -1))
        if len(x) > 1:
            x_p = self.emb_layers[1].lookup(x[1])
            x_in = T.concatenate([x_in, x_p], axis=3)

        return self.emb_layers[len(self.x)].dot(x_in)

    def hidden_layer_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        return self.hidden_layers.forward(x)

    def output_layer_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: n_words, 2D: batch * n_prds, 3D: n_labels; log probability of a label
        """
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = x.dimshuffle(1, 0, 2)
        return self.layers[-1].forward(x)
