import theano
import theano.tensor as T

import rnn_pasa_model
from utils import relu, tanh


def set_model(argv, emb, vocab):
    c = T.imatrix('c')
    l = T.ivector('l')
    n_words = T.iscalar('n_words')

    """ Set the classifier parameters"""
    window = argv.window + 1
    opt = argv.opt
    lr = argv.lr
    init_emb = emb
    dim_emb = argv.dim_emb if emb is None else len(emb[0])
    dim_hidden = argv.dim_hidden
    n_vocab = vocab.size()
    L2_reg = argv.reg
    unit = argv.unit
    activation = relu if argv.activation == 'relu' else tanh

    model = rnn_pasa_model.Model(c=c, l=l, n_words=n_words, window=window, opt=opt, lr=lr, init_emb=init_emb, dim_emb=dim_emb,
                                 dim_hidden=dim_hidden, n_vocab=n_vocab, L2_reg=L2_reg, unit=unit, activation=activation)
    return model


def set_train_f(model, tr_samples):
    index = T.iscalar('index')
    bos = T.iscalar('bos')
    eos = T.iscalar('eos')

    train_f = theano.function(inputs=[index, bos, eos],
                              outputs=[model.x, model.labels],
#                              updates=model.update,
                              givens={
                                  model.tr_inputs[0]: tr_samples[0][bos: eos],
                                  model.tr_inputs[1]: tr_samples[1][bos: eos],
                                  model.tr_inputs[2]: tr_samples[2][index],
                              }
                              )
    return train_f
