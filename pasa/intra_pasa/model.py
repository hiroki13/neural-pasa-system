import numpy as np
import theano
import theano.tensor as T

from ..utils.io_utils import say
from ..nn.rnn import GRU, ConnectedLayer
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
        x = self.embedding_layer(x, batch_size, n_words)
        self.x = x
        h = self.mid_layers(x, batch_size, dim_h)
        h = self.output_layer(h)

        ###########
        # Outputs #
        ###########
        layer = self.layers[-1]
        self.y_gold = y.reshape((batch_size, n_words))
        self.y_pred = layer.decode(h)
        self.p_y = layer.get_y_prob(h, self.y_gold.dimshuffle((1, 0)))

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(self.p_y, reg)
        self.update = self.optimize(opt, self.cost, lr)

    def set_layers(self, unit, n_vocab, init_emb, n_in, n_fin, n_h, n_y, n_layers):
        ###################
        # Embedding layer #
        ###################
        self.layers.append(EmbeddingLayer(n_vocab, n_in, init_emb))

        ##############
        # Mid layers #
        ##############
        if unit.lower() == 'gru':
            layer = GRU

        for i in xrange(n_layers):
            if i == 0:
                self.layers.append(ConnectedLayer(n_i=n_fin, n_h=n_h))
            self.layers.append(layer(n_i=n_h, n_h=n_h))

        #################
        # Output layers #
        #################
        if self.argv.output_layer == 0:
            self.layers.append(Layer(n_i=n_h, n_labels=n_y))
        elif self.argv.output_layer == 1:
            self.layers.append(MEMMLayer(n_i=n_h, n_labels=n_y))
        else:
            self.layers.append(CRFLayer(n_i=n_h, n_labels=n_y))

        say('No. of rnn layers: %d\n' % (len(self.layers)-3))

    def set_params(self):
        for l in self.layers:
            self.params += l.params
        say("No. of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def embedding_layer(self, x, batch, n_words):
        """
        :param x: 1D: batch * n_words, 2D: 5 + window + 1; elem=word_id
        :return: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        """
        return self.layers[0].lookup(x).reshape((batch, n_words, -1))

    def mid_layers(self, x, batch, dim_h):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_in (dim_emb * (5 + window + 1))
        :return: 1D: n_words, 2D: batch, 3D: dim_h
        """
        for i in xrange(1, len(self.layers)-2):
            # h0: 1D: batch, 2D: n_h
            if i == 1:
                layer = self.layers[i]
                x = layer.dot(x.dimshuffle(1, 0, 2))
                h0 = T.zeros((batch, dim_h), dtype=theano.config.floatX)
            else:
                x = (h + x)[::-1]
                h0 = x[0]
            layer = self.layers[i+1]
            # 1D: n_words, 2D: batch, 3D n_h
            h = layer.forward_all(x, h0)
        return x + h

    def output_layer(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h: 1D: n_words, 2D: batch, 3D: dim_h
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


class MultiSeqModel(Model):

    def __init__(self, argv, emb, vocab_word, vocab_label):

        super(MultiSeqModel, self).__init__(argv, emb, vocab_word, vocab_label)

    def compile(self, x, y, n_words, n_prds):
        argv = self.argv
        init_emb = self.emb

        ###################
        # Input variables #
        ###################
        self.inputs = [x, y, n_words, n_prds]

        ##############
        # Dimensions #
        ##############
        self.batch_size = x.shape[0] / n_words
        window = 5 + argv.window + 1
        dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        dim_in = dim_emb * window
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
        n_layers = argv.layer

        ################
        # Architecture #
        ################
        x = self.embedding_layer(x, n_words, n_vocab, dim_emb, dim_in, init_emb)
        x, h = self.mid_layers(x, dim_in, dim_h, unit, n_layers)
        h, layer = self.output_layer(x, h, dim_h, dim_out, n_layers)

        ###########
        # Outputs #
        ###########
        self.y_gold = y.reshape((self.batch_size, n_words))
        p_y = layer.get_y_prob(h, self.y_gold.dimshuffle((1, 0)))
        self.y_pred = layer.memm(h)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(p_y, reg)
        self.update = self.optimize(opt, self.cost, lr)

