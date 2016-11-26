import numpy as np
import theano
import theano.tensor as T

from ..utils.io_utils import say
from ..nn.rnn import RNNLayers, GridObliqueNetwork, ConnectedLayer, BiRNNLayers
from ..nn.nn_utils import L2_sqr
from ..nn.optimizers import ada_grad, ada_delta, adam, sgd
from ..nn.seq_labeling import Layer, MEMMLayer, CRFLayer, MixedLayer
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
        self.hidden_reps = None
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

    def compile(self, variables):
        argv = self.argv
        x_w, x_p, y = variables

        ###################
        # Input variables #
        ###################
        self.inputs = [x_w, x_p, y]

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(self.emb)
        self.set_params()

        ############
        # Networks #
        ############
        x = self.emb_layer_forward(x_w, x_p)
        h = self.hidden_layer_forward(x)
        o = self.output_layer_forward(h)

        ###########
        # Outputs #
        ###########
        self.y_gold = y
        self.y_pred = self.output_layer.decode(o)
        self.y_prob = o.dimshuffle(1, 0, 2)
        self.hidden_reps = h.dimshuffle(1, 0, 2)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(o=o, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self, init_emb):
        argv = self.argv
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb * (5 + argv.window) + dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels

        self.emb_layer = EmbeddingLayer(init_emb=init_emb, n_vocab=self.n_vocab, dim_emb=dim_emb,
                                        n_posit=2, dim_posit=dim_posit, fix=argv.fix)
        self.hidden_layers = RNNLayers(unit=argv.unit, depth=argv.layers, n_in=dim_in, n_h=dim_h)

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
            self.params.extend(l.params)
        say("No. of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def emb_layer_forward(self, x_w, x_p):
        """
        :param x_w: 1D: batch, 2D: n_words, 3D: 5 + window; word id
        :param x_p: 1D: batch, 2D: n_words; posit id
        :return: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        """
        x_w = self.emb_layer.forward_word(x_w).reshape((x_w.shape[0], x_w.shape[1], -1))
        x_p = self.emb_layer.forward_posit(x_p)
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
        :return: 1D: n_words, 2D: batch, 3D: n_labels
        """
        h = self.layers[-1].forward(x)
        if (len(self.layers) - 3) % 2 == 0:
            h = h[::-1]
        return h

    def objective_f(self, o, reg):
        p_y = self.output_layer.get_y_prob(o, self.y_gold.dimshuffle((1, 0)))
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


class GridModel(Model):

    def __init__(self, argv, emb, n_vocab, n_labels):
        super(GridModel, self).__init__(argv, emb, n_vocab, n_labels)
        self.emb_connected_layer = None

    def compile(self, variables):
        argv = self.argv
        # x_w: 1D: batch, 2D: n_prds, 3D: n_words, 4D: 5+window; word id
        # x_p: 1D: batch, 2D: n_prds, 3D: n_words; posit id
        # y: 1D: batch, 2D: n_prds, 3D: n_words; elem=label id
        x_w, x_p, y = variables
        self.inputs = [x_w, x_p, y]

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(self.emb)
        self.set_params()

        x = self.emb_layer_forward(x_w, x_p)
        h = self.hidden_layer_forward(x)
        h = self.output_layer_forward(h)

        self.y_pred = self.output_layer.decode(h)
        self.y_gold = y.reshape(self.y_pred.shape)
        self.y_prob = h.dimshuffle(1, 0, 2)

        self.nll, self.cost = self.objective_f(o=h, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self, init_emb):
        argv = self.argv
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb * (5 + argv.window) + dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels

        self.emb_layer = EmbeddingLayer(init_emb=init_emb, n_vocab=self.n_vocab, dim_emb=dim_emb,
                                        n_posit=2, dim_posit=dim_posit, fix=argv.fix)
        self.emb_connected_layer = ConnectedLayer(n_i=dim_in, n_h=dim_h)
        self.hidden_layers = GridObliqueNetwork(unit=argv.unit, depth=argv.layers, n_in=dim_h, n_h=dim_h)
        self.output_layer = Layer(n_i=dim_h, n_labels=dim_out)

        self.layers.append(self.emb_layer)
        self.layers.append(self.emb_connected_layer)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)
        say('No. of rnn layers: %d\n' % (len(self.layers)-3))

    def emb_layer_forward(self, x_w, x_p):
        """
        :param x_w: 1D: batch, 2D: n_prds, 3D: n_words, 4D: 5+window; word id
        :param x_p: 1D: batch, 2D: n_prds, 3D: n_words; 0/1
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_in
        """
        x_w = self.emb_layer.forward_word(x_w).reshape((x_w.shape[0], x_w.shape[1], x_w.shape[2], -1))
        x_p = self.emb_layer.forward_posit(x_p)
        return T.concatenate([x_w, x_p], axis=3)

    def hidden_layer_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_in
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        h = self.emb_connected_layer.dot(x)
        return self.hidden_layers.forward(h)

    def output_layer_forward(self, x):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: n_words, 2D: batch * n_prds, 3D: n_labels; log probability of a label
        """
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = x.dimshuffle(1, 0, 2)
        return self.layers[-1].forward(x)


class MixedModel(Model):

    def __init__(self, argv, emb, n_vocab, n_labels):
        super(MixedModel, self).__init__(argv, emb, n_vocab, n_labels)
        self.emb_h_layer = None
        self.emb_o_layer = None

    def compile(self, variables):
        argv = self.argv
        # x_w: 1D: batch, 2D: n_words; word id
        # x_p: 1D: batch, 2D: n_words; 0/1
        # x_m: 1D: batch, 2D: n_prds; prd index
        # y: 1D: batch, 2D: n_prds, 3D: n_words; elem=label id
        x_w, x_p, x_m, y = variables
        self.inputs = [x_w, x_p, x_m, y]

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(self.emb)
        self.set_params()

        x = self.emb_layer_forward(x_w, x_p, x_m)
        h = self.hidden_layer_forward(x)
        # 1D: n_words, 2D: batch * n_prds, 3D: n_labels
        h = self.output_layer_forward(h)

        # 1D: batch * n_prds, 2D: n_words; label id
        self.y_gold = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))
        # 1D: batch * n_prds, 2D: n_words; label id
        self.y_pred = self.output_layer.decode(h).reshape(self.y_gold.shape)
        self.y_prob = h.dimshuffle(1, 0, 2)

        self.nll, self.cost = self.objective_f(o=h, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self, init_emb):
        argv = self.argv
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb + dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels

        self.emb_layer = EmbeddingLayer(init_emb=init_emb, n_vocab=self.n_vocab, dim_emb=dim_emb,
                                        n_posit=2, dim_posit=dim_posit, fix=argv.fix)
        self.emb_h_layer = BiRNNLayers(unit=argv.unit, depth=argv.layers, n_in=dim_in, n_h=dim_h)
        self.emb_o_layer = MixedLayer(n_i=dim_h)
        self.hidden_layers = RNNLayers(unit=argv.unit, depth=argv.layers, n_in=dim_h, n_h=dim_h)
        self.output_layer = Layer(n_i=dim_h, n_labels=dim_out)

        self.layers.append(self.emb_layer)
        self.layers.extend(self.emb_h_layer.layers)
        self.layers.append(self.emb_o_layer)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.output_layer)
        say('No. of rnn layers: %d\n' % (len(self.layers)-7))

    def emb_layer_forward(self, x_w, x_p, x_m):
        """
        :param x_w: 1D: batch, 2D: n_words; word id
        :param x_p: 1D: batch, 2D: n_words; 0/1
        :return: 1D: batch * n_prds, 2D: n_words, 3D: dim_h
        """
        x_w = self.emb_layer.forward_word(x_w).reshape((x_w.shape[0], x_w.shape[1], -1))
        x_p = self.emb_layer.forward_posit(x_p)
        x = T.concatenate([x_w, x_p], axis=2)
        h = self.emb_h_layer.forward(x).dimshuffle(1, 0, 2)
        return self.emb_o_layer.forward(h, x_m)

    def output_layer_forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels
        """
        h = self.layers[-1].forward(x)
#        if (len(self.layers) - 7) % 2 == 0:
#            h = h[::-1]
        return h

    def objective_f(self, o, reg):
        p_y = self.output_layer.get_y_prob(o, self.y_gold.dimshuffle(1, 0))
        nll = - T.mean(p_y)
        cost = nll + reg * L2_sqr(self.params) / 2.
        return nll, cost


class StackingModel(GridModel):

    def __init__(self, argv, emb, n_vocab, n_labels):
        super(StackingModel, self).__init__(argv, emb, n_vocab, n_labels)
        self.hidden_connected_layer = None

    def compile(self, variables):
        argv = self.argv
        x_w, x_p, x_h, x_l, y = variables

        self.inputs = [x_w, x_p, x_h, x_l, y]

        self.dropout = theano.shared(np.float32(argv.dropout).astype(theano.config.floatX))
        self.set_layers(self.emb)
        self.set_params()

        x = self.emb_layer_forward(x_w, x_p)
        h = self.hidden_layer_forward(x, x_h, x_l)
        o = self.output_layer_forward(h)

        self.y_pred = self.output_layer.decode(o)
        self.y_gold = y.reshape(self.y_pred.shape)
        self.y_prob = o.dimshuffle(1, 0, 2)

        self.nll, self.cost = self.objective_f(o=o, reg=argv.reg)
        self.update = self.optimize(cost=self.cost, opt=argv.opt, lr=argv.lr)

    def set_layers(self, init_emb):
        argv = self.argv

        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_posit = argv.dim_posit
        dim_in = dim_emb * (5 + argv.window) + dim_posit
        dim_h = argv.dim_hidden
        dim_out = self.n_labels

        self.emb_layer = EmbeddingLayer(init_emb=init_emb, n_vocab=self.n_vocab, dim_emb=dim_emb,
                                        n_posit=2, dim_posit=dim_posit, fix=argv.fix)
        self.emb_connected_layer = ConnectedLayer(n_i=dim_in, n_h=dim_h)
        self.hidden_layers = GridObliqueNetwork(unit=argv.unit, depth=argv.layers, n_in=dim_h, n_h=dim_h)
        self.hidden_connected_layer = ConnectedLayer(n_i=dim_h*2+dim_out, n_h=dim_h)
        self.output_layer = Layer(n_i=dim_h, n_labels=dim_out)

        self.layers.append(self.emb_layer)
        self.layers.append(self.emb_connected_layer)
        self.layers.extend(self.hidden_layers.layers)
        self.layers.append(self.hidden_connected_layer)
        self.layers.append(self.output_layer)
        say('No. of rnn layers: %d\n' % (len(self.layers)-3))

    def hidden_layer_forward(self, x, x_h, x_l):
        """
        :param x: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        :return: 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        """
        h = self.hidden_layers.forward(x)
        h = T.concatenate([h, x_h, x_l], axis=3)
        return self.hidden_connected_layer.dot(h)

