import sys
import time
import gzip
import cPickle as pickle

import theano
import theano.tensor as T

from model import Model
from eval import Eval


class Decoder(object):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        self.argv = argv
        self.emb = emb
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.mp = True if argv.attention else False

        self.model = None
        self.train = None
        self.predict = None

    def set_model(self):
        # x: 1D: batch * n_words, 2D: 5 + window + 1; elem=word_id
        # y: 1D: batch * n_cands; elem=label
        x = T.imatrix('x')
        y = T.ivector('y')
        n_words = T.iscalar('n_words')
        n_prds = T.iscalar('n_prds')

        self.model = Model(argv=self.argv, emb=self.emb,
                           vocab_word=self.vocab_word, vocab_label=self.vocab_label)

        if self.mp:
            self.model.compile(x, y, n_words, n_prds)
        else:
            self.model.compile(x, y, n_words)

    def set_train_f(self, samples):
        index = T.iscalar('index')
        bos = T.iscalar('bos')
        eos = T.iscalar('eos')

        model = self.model

        if self.mp:
            self.train = theano.function(inputs=[index, bos, eos],
                                         outputs=[model.y_pred, model.y_gold, model.nll],
                                         updates=model.update,
                                         givens={
                                             model.inputs[0]: samples[0][bos: eos],
                                             model.inputs[1]: samples[1][bos: eos],
                                             model.inputs[2]: samples[2][index],
                                             model.inputs[3]: samples[3][index],
                                         }
                                         )
        else:
            self.train = theano.function(inputs=[index, bos, eos],
                                         outputs=[model.y_pred, model.y_gold, model.nll],
                                         updates=model.update,
                                         givens={
                                             model.inputs[0]: samples[0][bos: eos],
                                             model.inputs[1]: samples[1][bos: eos],
                                             model.inputs[2]: samples[2][index],
                                         }
                                         )

    def set_train_online_f(self):
        model = self.model
        self.train = theano.function(inputs=model.inputs,
                                     outputs=[model.y_pred, model.y_gold, model.nll],
                                     updates=model.update,
                                     )

    def set_predict_f(self):
        model = self.model
        self.predict = theano.function(inputs=model.inputs,
                                       outputs=model.y_pred,
                                       on_unused_input='ignore'
                                       )

    def predict_all(self, samples):
        """
        :param samples: 1D: n_sents: Sample
        """
        pred_eval = Eval()
        start = time.time()
        self.model.dropout.set_value(0.0)

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                continue

            if self.mp:
                results_sys = self.predict(sample.x, sample.y, sample.n_words, sample.n_prds)
            else:
                results_sys = self.predict(sample.x, sample.y, sample.n_words)

            pred_eval.update_results(results_sys, sample.label_ids)

        print '\tTime: %f' % (time.time() - start)
        pred_eval.show_results()

        return pred_eval.all_f1

    def output_results(self, fn, samples):
        ###########
        # Predict #
        ###########
        results = []
        for index, sample in enumerate(samples):

            if sample.n_prds == 0:
                results.append([])
                continue

            if self.mp:
                results_sys = self.predict(sample.x, sample.y, sample.n_words, sample.n_prds)
            else:
                results_sys = self.predict(sample.x, sample.y, sample.n_words)
            results.append(results_sys)

        assert len(samples) == len(results)

        ##########
        # Output #
        ##########
        vocab_word = self.vocab_word
        vocab_label = self.vocab_label

        with open(fn, 'w') as fout:
            for sent_index in xrange(len(samples)):
                sample = samples[sent_index]
                result = results[sent_index]
                g_result = sample.label_ids

                #################
                # Raw sentences #
                #################
                text = 'SENT-%d ' % (sent_index + 1)
                for i, w in enumerate(sample.word_ids):
                    text += '%d:%s ' % (i, vocab_word.get_word(w))
                print >> fout, text.encode('utf-8')

                ################
                # PASA results #
                ################
                for i in xrange(len(result)):
                    r = result[i]
                    g_r = g_result[i]

                    prd_index = sample.prd_indices[i]
                    prd_id = sample.word_ids[prd_index]

                    ########
                    # Gold #
                    ########
                    text = 'GOLD-%d %d:%s\t' % (i+1, prd_index, vocab_word.get_word(prd_id))
                    for w_index, label in enumerate(g_r):
                        if 0 < label < 4:
                            w_id = sample.word_ids[w_index]
                            text += '%s:%d:%s ' % (vocab_label.get_word(label), w_index, vocab_word.get_word(w_id))
                    print >> fout, text.encode('utf-8')

                    ##########
                    # System #
                    ##########
                    text = 'PRED-%d %d:%s\t' % (i+1, prd_index, vocab_word.get_word(prd_id))
                    for w_index, label in enumerate(r):
                        if 0 < label < 4:
                            w_id = sample.word_ids[w_index]
                            text += '%s:%d:%s ' % (vocab_label.get_word(label), w_index, vocab_word.get_word(w_id))
                    print >> fout, text.encode('utf-8')

                print >> fout

    def save_model(self, path):
        if not path.endswith(".pkl.gz"):
            path += ".gz" if path.endswith(".pkl") else ".pkl.gz"

        with gzip.open(path, "w") as fout:
            pickle.dump(
                {
                    'argv': self.argv,
                    'model': self.model,
                },
                fout,
                protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, argv):
        with gzip.open(argv.load) as fin:
            data = pickle.load(fin)
            self.argv = data['argv']
            self.model = data['model']


class MPDecoder(Decoder):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        super(MPDecoder, self).__init__(argv, emb, vocab_word, vocab_label)

    def set_model(self):
        argv = self.argv

        ###################
        # Input variables #
        ###################
        # x: 1D: batch * n_words, 2D: window + 1; elem=word_id
        # y: 1D: batch * n_cands; elem=label
        x = T.imatrix('x')
        y = T.ivector('y')
        n_words = T.iscalar('n_words')
        n_prds = T.iscalar('n_prds')

        ###################
        # Hyperparameters #
        ###################
        dropout = argv.dropout
        window = 5 + argv.window + 1
        opt = argv.opt
        lr = argv.lr
        init_emb = self.emb
        n_in = argv.dim_emb if self.emb is None else len(self.emb[0])
        n_h = argv.dim_hidden
        n_y = self.vocab_label.size()
        n_vocab = self.vocab_word.size()
        L2_reg = argv.reg
        unit = argv.unit
        n_layers = argv.layer

        self.model = Model(x=x, y=y, n_words=n_words, n_prds=n_prds, n_vocab=n_vocab, init_emb=init_emb,
                           n_in=n_in, n_h=n_h, n_y=n_y, window=window, unit=unit, opt=opt, lr=lr, dropout=dropout,
                           L2_reg=L2_reg, n_layers=n_layers)

    def set_train_f(self, samples):
        index = T.iscalar('index')
        bos = T.iscalar('bos')
        eos = T.iscalar('eos')

        model = self.model
        self.train = theano.function(inputs=[index, bos, eos],
                                     outputs=[model.y_pred, model.y_gold, model.nll],
                                     updates=model.update,
                                     givens={
                                         model.inputs[0]: samples[0][bos: eos],
                                         model.inputs[1]: samples[1][bos: eos],
                                         model.inputs[2]: samples[2][index],
                                         model.inputs[3]: samples[3][index],
                                     }
                                     )

    def set_predict_f(self):
        model = self.model
        self.predict = theano.function(inputs=model.inputs,
                                       outputs=model.y_pred,
                                       on_unused_input='ignore'
                                       )

    def predict_all(self, samples):
        """
        :param samples: 1D: n_sents: Sample
        """
        pred_eval = Eval()
        start = time.time()
        self.model.dropout.set_value(0.0)

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                continue

            results_sys = self.predict(sample.x, sample.y, sample.n_words)
            pred_eval.update_results(results_sys, sample.label_ids)

        print '\tTime: %f' % (time.time() - start)
        pred_eval.show_results()

        return pred_eval.all_f1

