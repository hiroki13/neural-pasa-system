import numpy as np
import theano
import theano.tensor as T

from ..nn.utils import L2_sqr, relu
from ..nn.optimizers import ada_grad, ada_delta, adam, sgd
from ..nn import gru, lstm, cnn
from ..nn.crf import CRFLayer, Layer
from ..nn.embedding import EmbeddingLayer
from ..nn.attention import AttentionLayer


class Model(object):

    def __init__(self, argv, emb, vocab_word, vocab_label):

        self.argv = argv
        self.emb = emb
        self.n_vocab = vocab_word.size()
        self.n_labels = vocab_label.size()
        self.batch_size = None
        self.dropout = None

        ###################
        # Input variables #
        ###################
        self.inputs = None

        ####################
        # Output variables #
        ####################
        self.y_gold = None
        self.y_pred = None
        self.nll = None
        self.cost = None

        ##############
        # Parameters #
        ##############
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
        attention = argv.attention
        lr = argv.lr
        reg = argv.reg
        opt = argv.opt
        unit = argv.unit
        pooling = argv.pooling
        n_layers = argv.layer

        ################
        # Architecture #
        ################
        x = self.embedding_layer(x, n_words, n_vocab, dim_emb, dim_in, init_emb)
        x, h = self.intermediate_layers(x, dim_in, dim_h, unit, n_layers)
        h, layer = self.output_layer(x, h, dim_h, dim_out, n_layers, n_prds, attention, pooling)

        ###########
        # Outputs #
        ###########
        self.y_gold = y.reshape((self.batch_size, n_words))
        p_y = layer.y_prob(h, self.y_gold.dimshuffle((1, 0)))
        self.y_pred = layer.decode(h)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(p_y, reg)
        self.update = self.optimize(opt, self.cost, lr)

    def embedding_layer(self, x, n_words, n_vocab, dim_emb, dim_in, init_emb):
        layer = EmbeddingLayer(n_vocab, dim_emb, init_emb)
        self.params += layer.params

        x_in = layer.lookup(x)

        return x_in.reshape((self.batch_size, n_words, dim_in))

    def intermediate_layers(self, x, dim_in, dim_h, unit, n_layers):
        if unit.lower() == 'lstm':
            layers = lstm.layers
        else:
            layers = gru.set_layers

        params, h, x = layers(x=x, batch=self.batch_size, n_fin=dim_in, n_h=dim_h,
                              dropout=self.dropout, n_layers=n_layers)
        self.params += params

        return x, h

    def output_layer(self, x, h, dim_h, dim_out, n_layers, n_prds, attention, pooling):
        #######################
        # Attention mechanism #
        #######################
        if attention == 1:
            layer = AttentionLayer(n_h=dim_h)
            h = layer.multi_prd_attention(h=h, n_prds=n_prds)
            self.params += layer.params
        elif attention == 2:
            if pooling == 'max':
                p = T.max
            else:
                p = T.mean
            layer = cnn.Layer(n_h=dim_h, pooling=p)
            h = layer.convolution(h=h, n_prds=n_prds)
            self.params += layer.params

        ################
        # Output layer #
        ################
        if self.argv.output_layer == 0:
            layer = Layer(n_i=dim_h * 2, n_h=dim_out)
        else:
            layer = CRFLayer(n_i=dim_h * 2, n_h=dim_out)
        self.params += layer.params

        h = relu(T.dot(T.concatenate([x, h], 2), layer.W))

        if n_layers % 2 == 0:
            h = h[::-1]

        return h, layer

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
        x, h = self.intermediate_layers(x, dim_in, dim_h, unit, n_layers)
        h, layer = self.output_layer(x, h, dim_h, dim_out, n_layers)

        ###########
        # Outputs #
        ###########
        self.y_gold = y.reshape((self.batch_size, n_words))
        p_y = layer.y_prob(h, self.y_gold.dimshuffle((1, 0)))
        self.y_pred = layer.memm(h)

        ############
        # Training #
        ############
        self.nll, self.cost = self.objective_f(p_y, reg)
        self.update = self.optimize(opt, self.cost, lr)

