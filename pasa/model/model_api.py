import sys
import time
import gzip
import math
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T

from model import Model, RankingModel
from ..utils.io_utils import say
from ..utils.eval import Eval


class ModelAPI(object):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        self.argv = argv
        self.emb = emb
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label

        self.model = None
        self.train = None
        self.predict = None

    def compile(self, train_sample_shared=None):
        say('\n\nBuilding a model API...\n')
        self.set_model()
        self.compile_model()
        self.set_train_f(train_sample_shared)
        self.set_predict_f()

    def set_model(self):
        self.model = Model(argv=self.argv,
                           emb=self.emb,
                           n_vocab=self.vocab_word.size(),
                           n_labels=self.vocab_label.size())

    def compile_model(self):
        # x: 1D: batch * n_words, 2D: 5 + window + 1; elem=word id
        # y: 1D: batch * n_words; elem=label id
        self.model.compile(x_w=T.imatrix('x_w'),
                           x_p=T.ivector('x_p'),
                           y=T.ivector('y'),
                           n_words=T.iscalar('n_words'))

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
                                         model.inputs[2]: samples[2][bos: eos],
                                         model.inputs[3]: samples[3][index],
                                     }
                                     )

    def set_predict_f(self):
        model = self.model
        self.predict = theano.function(inputs=model.inputs,
                                       outputs=model.y_pred,
                                       on_unused_input='ignore'
                                       )

    def train_all(self, argv, train_batch_index, dev_samples, test_samples, untrainable_emb=None):
        say('\n\nTRAINING START\n\n')

        n_train_batches = len(train_batch_index)
        tr_indices = range(n_train_batches)

        f1_history = {}
        best_dev_f1 = -1.

        for epoch in xrange(argv.epoch):
            dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
            self.model.dropout.set_value(dropout_p)

            say('\nEpoch: %d\n' % (epoch + 1))
            print '  TRAIN\n\t',

            self.train_each(tr_indices, train_batch_index)

            ###############
            # Development #
            ###############
            if untrainable_emb is not None:
                trainable_emb = self.model.emb_layer.word_emb.get_value(True)
                self.model.emb_layer.word_emb.set_value(np.r_[trainable_emb, untrainable_emb])

            update = False
            if argv.dev_data:
                print '\n  DEV\n\t',
                dev_f1 = self.predict_all(dev_samples)
                if best_dev_f1 < dev_f1:
                    best_dev_f1 = dev_f1
                    f1_history[epoch+1] = [best_dev_f1]
                    update = True

                    if argv.save:
                        self.save_params('params.intra.layers-%d.window-%d.reg-%f' %
                                         (argv.layers, argv.window, argv.reg))
                        self.save_config('config.intra.layers-%d.window-%d.reg-%f' %
                                         (argv.layers, argv.window, argv.reg))

                    if argv.result:
                        self.output_results('result.dev.txt', dev_samples)

            ########
            # Test #
            ########
            if argv.test_data:
                print '\n  TEST\n\t',
                test_f1 = self.predict_all(test_samples)
                if update:
                    if epoch+1 in f1_history:
                        f1_history[epoch+1].append(test_f1)
                    else:
                        f1_history[epoch+1] = [test_f1]

            if untrainable_emb is not None:
                self.model.emb_layer.word_emb.set_value(trainable_emb)

            ###########
            # Results #
            ###########
            say('\n\n\tF1 HISTORY')
            for k, v in sorted(f1_history.items()):
                if len(v) == 2:
                    say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
                else:
                    say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
            say('\n\n')

    def train_each(self, tr_indices, train_batch_index):
        np.random.shuffle(tr_indices)
        train_eval = Eval()
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            batch_range = train_batch_index[b_index]
            result_sys, result_gold, nll = self.train(index=b_index, bos=batch_range[0], eos=batch_range[1])

            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_results()

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

            results_sys = self.predict(sample.x_w, sample.x_p, sample.y, sample.n_words)
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

            results_sys = self.predict(sample.x_w, sample.x_p, sample.y, sample.n_words)
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

    def save_params(self, path):
        if not path.endswith(".pkl.gz"):
            path += ".gz" if path.endswith(".pkl") else ".pkl.gz"
        with gzip.open(path, "w") as fout:
            pickle.dump([l.params for l in self.model.layers], fout,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def save_config(self, path):
        if not path.endswith(".pkl.gz"):
            path += ".gz" if path.endswith(".pkl") else ".pkl.gz"
        with gzip.open(path, "w") as fout:
            pickle.dump(self.argv, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        with gzip.open(path) as fin:
            params = pickle.load(fin)
            assert len(self.model.layers) == len(params)

            for l, p in zip(self.model.layers, params):
                for p1, p2 in zip(l.params, p):
                    p1.set_value(p2.get_value(borrow=True))


class RankingModelAPI(ModelAPI):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        super(RankingModelAPI, self).__init__(argv, emb, vocab_word, vocab_label)

    def compile(self):
        say('\n\nBuilding a ranking model API...\n')
        self.set_model()
        self.compile_model()
        self.set_train_f()
        self.set_predict_f()

    def set_model(self):
        self.model = RankingModel(argv=self.argv,
                                  emb=self.emb,
                                  n_vocab=self.vocab_word.size(),
                                  n_labels=3)

    def compile_model(self):
        # x: 1D: batch * n_words, 2D: 5 + window; elem=word id
        # y: 1D: batch, 2D: n_labels (3); elem=label id
        self.model.compile(x_w=T.imatrix('x_w'),
                           x_p=T.ivector('x_p'),
                           y=T.imatrix('y'),
                           n_words=T.iscalar('n_words'))

    def set_train_f(self):
        model = self.model
        self.train = theano.function(inputs=model.inputs,
                                     outputs=[model.y_pred, model.y_gold, model.nll],
                                     updates=model.update,
                                     on_unused_input='ignore'
                                     )

    def train_all(self, argv, train_samples, dev_samples, test_samples, untrainable_emb=None):
        say('\n\nTRAINING START\n\n')

        n_train_batches = len(train_samples)
        tr_indices = range(n_train_batches)

        f1_history = {}
        best_dev_f1 = -1.

        for epoch in xrange(argv.epoch):
            dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
            self.model.dropout.set_value(dropout_p)

            say('\nEpoch: %d\n' % (epoch + 1))
            print '  TRAIN\n\t',

            self.train_each(tr_indices, train_samples)

            ###############
            # Development #
            ###############
            if untrainable_emb is not None:
                trainable_emb = self.model.emb_layer.word_emb.get_value(True)
                self.model.emb_layer.word_emb.set_value(np.r_[trainable_emb, untrainable_emb])

            update = False
            if argv.dev_data:
                print '\n  DEV\n\t',
                dev_f1 = self.predict_all(dev_samples)
                if best_dev_f1 < dev_f1:
                    best_dev_f1 = dev_f1
                    f1_history[epoch+1] = [best_dev_f1]
                    update = True

                    if argv.save:
                        self.save_params('params.intra.layers-%d.window-%d.reg-%f' %
                                         (argv.layers, argv.window, argv.reg))
                        self.save_config('config.intra.layers-%d.window-%d.reg-%f' %
                                         (argv.layers, argv.window, argv.reg))

                    if argv.result:
                        self.output_results('result.dev.txt', dev_samples)

            ########
            # Test #
            ########
            if argv.test_data:
                print '\n  TEST\n\t',
                test_f1 = self.predict_all(test_samples)
                if update:
                    if epoch+1 in f1_history:
                        f1_history[epoch+1].append(test_f1)
                    else:
                        f1_history[epoch+1] = [test_f1]

            if untrainable_emb is not None:
                self.model.emb_layer.word_emb.set_value(trainable_emb)

            ###########
            # Results #
            ###########
            say('\n\n\tF1 HISTORY')
            for k, v in sorted(f1_history.items()):
                if len(v) == 2:
                    say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
                else:
                    say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
            say('\n\n')

    def train_each(self, tr_indices, train_samples):
        np.random.shuffle(tr_indices)
        train_eval = Eval()
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            x_w, x_p, y, n_words = train_samples[b_index]
            result_sys, result_gold, nll = self.train(x_w, x_p, y, n_words)

            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_rank_results(result_sys, result_gold, n_words)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_results()

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

            results_sys = self.predict(sample.x_w, sample.x_p, sample.y, sample.n_words)
            pred_eval.update_rank_results(results_sys, sample.label_ids, sample.n_words)

        print '\tTime: %f' % (time.time() - start)
        pred_eval.show_results()

        return pred_eval.all_f1

