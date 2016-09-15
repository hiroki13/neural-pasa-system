import theano
import theano.tensor as T

import pasa_model


def set_model(argv, emb, vocab_word, vocab_label):
    x = T.imatrix('x')
    y = T.ivector('y')
    n_words = T.iscalar('n_words')
    n_prds = T.iscalar('n_prds')

    """ Set the classifier parameters"""
    window = 5 + argv.window + 1
    opt = argv.opt
    lr = argv.lr
    init_emb = emb
    dim_emb = argv.dim_emb if emb is None else len(emb[0])
    dim_hidden = argv.dim_hidden
    dim_out = vocab_label.size()
    n_vocab = vocab_word.size()
    L2_reg = argv.reg
    unit = argv.unit
    dropout = argv.dropout
    attention = argv.attention
    mp_cnn = argv.mp_cnn
    n_layers = argv.layer

    model = pasa_model.Model(x=x, y=y, n_words=n_words, n_prds=n_prds, window=window, opt=opt, lr=lr, init_emb=init_emb,
                             dim_emb=dim_emb, dim_hidden=dim_hidden, dim_out=dim_out, n_vocab=n_vocab, L2_reg=L2_reg,
                             unit=unit, dropout=dropout, attention=attention, mp_cnn=mp_cnn, n_layers=n_layers)
    return model


def set_train_f(model, tr_samples):
    index = T.iscalar('index')
    bos = T.iscalar('bos')
    eos = T.iscalar('eos')

    train_f = theano.function(inputs=[index, bos, eos],
                              outputs=[model.y_pred, model.y_reshaped, model.nll],
                              updates=model.update,
                              givens={
                                  model.tr_inputs[0]: tr_samples[0][bos: eos],
                                  model.tr_inputs[1]: tr_samples[1][bos: eos],
                                  model.tr_inputs[2]: tr_samples[2][index],
                                  model.tr_inputs[3]: tr_samples[3][index],
                              }
                              )
    return train_f


def set_predict_f(model):
    return theano.function(inputs=model.inputs,
                           outputs=[model.y_pred, model.y_reshaped],
                           )


def set_pred_f(model, samples):
    index = T.iscalar('index')
    bos = T.iscalar('bos')
    eos = T.iscalar('eos')

    pred_f = theano.function(inputs=[index, bos, eos],
                             outputs=[model.y_pred, model.y_reshaped],
                             givens={
                                 model.pr_inputs[0]: samples[0][bos: eos],
                                 model.pr_inputs[1]: samples[1][bos: eos],
                                 model.pr_inputs[2]: samples[2][index],
                                 model.pr_inputs[3]: samples[3][index],
                             }
                             )
    return pred_f
