import sys
import os
import time
import gzip
import math
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T

from model import Model, RankingModel, RerankingModel, GridModel
from ..utils.io_utils import say, move_data
from ..utils.eval import Eval
from ..decoder.decoder import Decoder
from ..utils.nbl_factory import NBestListFactory


class ModelAPI(object):

    def __init__(self, argv):
        self.argv = argv
        self.emb = None
        self.output_fn = None
        self.output_dir = None
        self.vocab_word = None
        self.vocab_label = None

        self.model = None
        self.decoder = None
        self.train = None
        self.predict = None
        self.predict_hidden_rep = None

    def compile(self, vocab_word, vocab_label, init_emb=None):
        say('\n\nBuilding a model API...\n')
        self.emb = init_emb
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.set_model()
        self.set_decoder()
        self.set_output_path()

    def set_output_path(self):
        self.output_fn = self._set_output_fn()
        self.output_dir = self._set_output_dir()

    def set_model(self):
        self.model = Model(argv=self.argv,
                           emb=self.emb,
                           n_vocab=self.vocab_word.size(),
                           n_labels=self.vocab_label.size())
        self.compile_model()

    def compile_model(self):
        # x_w: 1D: batch, 2D: n_words, 3D: 5 + window; word id
        # x_p: 1D: batch, 2D: n_words; posit id
        # y: 1D: batch, 2D: n_words; label id
        self.model.compile(x_w=T.itensor3('x_w'),
                           x_p=T.imatrix('x_p'),
                           y=T.imatrix('y'))

    def set_decoder(self):
        self.decoder = Decoder()

    def set_train_f(self):
        model = self.model
        self.train = theano.function(inputs=model.inputs,
                                     outputs=[model.y_pred, model.y_gold, model.nll],
                                     updates=model.update
                                     )

    def set_predict_f(self):
        model = self.model
        self.predict = theano.function(inputs=model.inputs,
                                       outputs=model.y_prob,
                                       on_unused_input='ignore'
                                       )

    def set_hidden_rep_f(self):
        model = self.model
        self.predict_hidden_rep = theano.function(inputs=model.inputs,
                                                  outputs=model.hidden_reps,
                                                  on_unused_input='ignore'
                                                  )

    def train_each(self, samples):
        tr_indices = range(len(samples))
        np.random.shuffle(tr_indices)
        train_eval = Eval()
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            x_w, x_p, y = samples[b_index]
            result_sys, result_gold, nll = self.train(x_w, x_p, y)
            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_results()

    def predict_all(self, samples):
        all_best_lists = []
        all_prob_lists = []
        start = time.time()

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                all_best_lists.append([])
                all_prob_lists.append([])
                continue

            prob_lists = self.predict(sample.x_w, sample.x_p, sample.y)
            best_list = self.decode_argmax(prob_lists=prob_lists, prd_indices=sample.prd_indices)
            all_best_lists.append(best_list)
            all_prob_lists.append(prob_lists)

        print '\tTime: %f' % (time.time() - start)
        return all_best_lists, all_prob_lists

    def output_hidden_rep(self, samples):
        all_hidden_reps = []
        start = time.time()

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                all_hidden_reps.append((sample, []))
                continue

            hidden_reps = self.predict(sample.x_w, sample.x_p, sample.y, sample.n_words)
            all_hidden_reps.append((sample, hidden_reps))

        print '\tTime: %f' % (time.time() - start)
        self._save_hidden_rep(self.output_fn, self.output_dir, all_hidden_reps)

    def decode_argmax(self, prob_lists, prd_indices):
        assert len(prob_lists) == len(prd_indices)
        return self.decoder.decode_argmax(prob_lists, prd_indices)

    @staticmethod
    def eval_all(results, samples):
        pred_eval = Eval()
        assert len(results) == len(samples)
        for result, sample in zip(results, samples):
            if len(result) == 0:
                continue
            pred_eval.update_results(batch_y_hat=result, batch_y=sample.label_ids)
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

    def _set_output_fn(self):
        argv = self.argv
        if argv.output_fn is None:
            return 'model-%s.layers-%d' % (argv.model, argv.layers)
        return argv.output_fn

    def _set_output_dir(self):
        argv = self.argv
        if argv.output_dir is not None and os.path.exists(argv.output_dir):
            return argv.output_dir
        if not os.path.exists('data'):
            os.mkdir('data')
        return 'data/'

    def save(self):
        self._save_params(self.output_fn, self.output_dir)
        self._save_config(self.output_fn, self.output_dir)

    def _save_params(self, fn, output_dir):
        fn = 'param.' + fn
        if not fn.endswith(".pkl.gz"):
            fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
        with gzip.open(fn, "w") as fout:
            pickle.dump([l.params for l in self.model.layers], fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        output_dir += 'param'
        move_data(fn, output_dir)

    def _save_config(self, fn, output_dir):
        fn = 'config.' + fn
        if not fn.endswith(".pkl.gz"):
            fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
        with gzip.open(fn, "w") as fout:
            pickle.dump(self.argv, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        output_dir += 'config'
        move_data(fn, output_dir)

    @staticmethod
    def _save_hidden_rep(fn, output_dir, data):
        fn = 'pretrained.' + fn
        if not fn.endswith(".pkl.gz"):
            fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
        with gzip.open(fn, "w") as fout:
            pickle.dump(data, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        output_dir += 'pretrained'
        move_data(fn, output_dir)

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
                                  n_labels=4)

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

    def train_each(self, tr_indices, samples):
        np.random.shuffle(tr_indices)
        train_eval = Eval()
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            x_w, x_p, y, n_words = samples[b_index]
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


class NBestModelAPI(ModelAPI):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        super(NBestModelAPI, self).__init__(argv, emb, vocab_word, vocab_label)
        self.nbl_factory = NBestListFactory(argv.n_best)

    def _set_output_fn(self):
        argv = self.argv
        if argv.output_fn is None:
            output_fn = 'model-%s.layers-%d.n_best-%d' % (argv.model, argv.layers, argv.n_best)
            if argv.sec is None:
                output_fn += '.all'
            else:
                output_fn += '.sec-%d' % argv.sec
            return output_fn
        return argv.output_fn

    def _set_output_dir(self):
        argv = self.argv
        if argv.output_dir is not None and os.path.exists(argv.output_dir):
            return argv.output_dir
        if not os.path.exists('data/rerank'):
            os.mkdir('data/rerank')
        return 'data/rerank/'

    def predict_n_best_lists(self, samples):
        _, results_prob = self.predict_all(samples)
        n_best_lists = self.create_n_best_lists(samples=samples, all_prob_lists=results_prob)
        self.eval_n_best_lists(samples, n_best_lists)
        return n_best_lists

    def create_n_best_lists(self, samples, all_prob_lists):
        return self.nbl_factory.create_n_best_lists(decoder=self.decoder,
                                                    samples=samples,
                                                    all_prob_lists=all_prob_lists)

    def eval_n_best_lists(self, samples, n_best_lists):
        self.nbl_factory.eval_n_best_list(samples, n_best_lists)

    @staticmethod
    def save_n_best_lists(fn, output_dir, n_best_lists):
        fn = 'list.' + fn
        if not fn.endswith(".pkl.gz"):
            fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
        with gzip.open(fn, "w") as fout:
            pickle.dump(n_best_lists, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        move_data(fn, output_dir)


class RerankingModelAPI(ModelAPI):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        super(RerankingModelAPI, self).__init__(argv, emb, vocab_word, vocab_label)

    def set_model(self):
        self.model = RerankingModel(argv=self.argv,
                                    emb=self.emb,
                                    n_vocab=self.vocab_word.size())

    def compile_model(self):
        # x_w: 1D: batch, 2D: n_prds, 3D: n_words, 4D: 5 + window; elem=word id
        # x_p: 1D: batch, 2D: n_prds, 3D: n_words; elem=posit id
        # x_l: 1D: batch, 2D: n_lists, 3D: n_prds, 4D: n_words; elem=label id
        # y: 1D: batch * n_words; elem=label id
        self.model.compile(x_w=T.itensor4('x_w'),
                           x_p=T.itensor3('x_p'),
                           x_l=T.itensor4('x_l'),
                           y=T.ivector('y'))

    def set_train_f(self, samples):
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

    def train_all(self, argv, train_samples, dev_samples, test_samples, untrainable_emb=None):
        say('\n\nTRAINING START\n\n')

        f1_history = {}
        best_dev_f1 = -1.
        for epoch in xrange(argv.epoch):
            dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
            self.model.dropout.set_value(dropout_p)

            say('\nEpoch: %d\n' % (epoch + 1))
            print '  TRAIN\n\t',

            self.train_each(train_samples)

            ###############
            # Development #
            ###############
            if untrainable_emb is not None:
                trainable_emb = self.model.emb_layer.word_emb.get_value(True)
                self.model.emb_layer.word_emb.set_value(np.r_[trainable_emb, untrainable_emb])

            update = False
            if argv.dev_data:
                print '\n  DEV\n\t',
                dev_results = self.predict_all(dev_samples)
                dev_f1 = self.eval_all(dev_results, dev_samples)
                if best_dev_f1 < dev_f1:
                    best_dev_f1 = dev_f1
                    f1_history[epoch+1] = [best_dev_f1]
                    update = True

                    if argv.save:
                        self.save()

            ########
            # Test #
            ########
            if argv.test_data:
                print '\n  TEST\n\t',
                test_results = self.predict_all(test_samples)
                test_f1 = self.eval_all(test_results, test_samples)
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

    def train_each(self, samples):
        tr_indices = range(len(samples))
        np.random.shuffle(tr_indices)
        train_eval = Eval()
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            x_w, x_p, x_l, oracle_y, y = samples[b_index]
            result_sys, result_gold, nll = self.train(x_w, x_p, x_l, oracle_y)
            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            y_hat = self.extract_labels(x_l, result_sys)
            train_eval.update_results(y_hat, y)
            train_eval.update_rerank_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_accuracy()
        train_eval.show_results()

    def predict_all(self, samples):
        all_best_lists = []
        all_prob_lists = []
        start = time.time()
        self.model.dropout.set_value(0.0)

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                all_best_lists.append([])
                all_prob_lists.append([])
                continue

            result_sys = self.predict([sample.x_w], [sample.x_p], [sample.x_l], [sample.y])
            best_list = self.extract_labels([sample.x_l], result_sys)
            all_best_lists.append(best_list)

        print '\tTime: %f' % (time.time() - start)
        return all_best_lists

    @staticmethod
    def extract_labels(labels, y_indices):
        """
        :param labels: 1D: batch, 2D: n_lists, 3D: n_prds, 4D: n_words; label id
        :param y_indices: 1D: batch; index
        :return: 1D: batch * n_prds, 2D: n_words; label id
        """
        assert len(labels) == len(y_indices), '%s\n%s' % (str(labels), str(y_indices))
        best_labels = []
        for n_list, index in zip(labels, y_indices):
            best_labels.extend(n_list[index])
        return best_labels

    def _set_output_fn(self):
        argv = self.argv
        if argv.output_fn is None:
            return 'model-%s.layers-%d' % (argv.model, argv.layers)
        return argv.output_fn

    def _set_output_dir(self):
        argv = self.argv
        if argv.output_dir is not None and os.path.exists(argv.output_dir):
            return argv.output_dir
        if not os.path.exists('data/rerank'):
            os.mkdir('data/rerank')
        return 'data/rerank/'


class GridModelAPI(ModelAPI):

    def __init__(self, argv, emb, vocab_word, vocab_label):
        super(GridModelAPI, self).__init__(argv, emb, vocab_word, vocab_label)

    def set_model(self):
        self.model = GridModel(argv=self.argv,
                               emb=self.emb,
                               n_vocab=self.vocab_word.size(),
                               n_labels=self.vocab_label.size())

    def compile_model(self):
        # x_w: 1D: batch, 2D: n_prds, 3D: n_words, 4D: 5 + window; elem=word id
        # x_p: 1D: batch, 2D: n_prds, 3D: n_words; elem=posit id
        # y: 1D: batch, 2D: n_prds, 3D: n_words; elem=label id
        self.model.compile(x_w=T.itensor4('x_w'),
                           x_p=T.itensor3('x_p'),
                           y=T.itensor3('y'))

    def set_train_f(self, samples):
        model = self.model
        self.train = theano.function(inputs=model.inputs,
                                     outputs=[model.y_pred, model.y_gold, model.nll],
                                     updates=model.update,
                                     )

    def set_predict_f(self):
        model = self.model
        self.predict = theano.function(inputs=model.inputs,
                                       outputs=model.y_prob,
                                       on_unused_input='ignore'
                                       )

    def train_all(self, argv, train_samples, dev_samples, test_samples, untrainable_emb=None):
        say('\n\nTRAINING START\n\n')

        f1_history = {}
        best_dev_f1 = -1.
        for epoch in xrange(argv.epoch):
            say('\nEpoch: %d\n' % (epoch + 1))
            print '  TRAIN\n\t',

            self.train_each(train_samples)

            ###############
            # Development #
            ###############
            if untrainable_emb is not None:
                trainable_emb = self.model.emb_layer.word_emb.get_value(True)
                self.model.emb_layer.word_emb.set_value(np.r_[trainable_emb, untrainable_emb])

            update = False
            if dev_samples:
                print '\n  DEV\n\t',
                dev_results, dev_results_prob = self.predict_all(dev_samples)
                dev_f1 = self.eval_all(dev_results, dev_samples)
                if best_dev_f1 < dev_f1:
                    best_dev_f1 = dev_f1
                    f1_history[epoch+1] = [best_dev_f1]
                    update = True

                    if argv.save:
                        self.save()

            ########
            # Test #
            ########
            if test_samples:
                print '\n  TEST\n\t',
                test_results, test_results_prob = self.predict_all(test_samples)
                test_f1 = self.eval_all(test_results, test_samples)
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

    def train_each(self, samples):
        tr_indices = range(len(samples))
        np.random.shuffle(tr_indices)
        train_eval = Eval()
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            x_w, x_p, y = samples[b_index]
            result_sys, result_gold, nll = self.train(x_w, x_p, y)
            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_results()

    def predict_all(self, samples):
        all_best_lists = []
        all_prob_lists = []
        start = time.time()

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                all_best_lists.append([])
                all_prob_lists.append([])
                continue

            prob_lists = self.predict([sample.x_w], [sample.x_p], [sample.y])
            best_list = self.decode_argmax(prob_lists=prob_lists, prd_indices=sample.prd_indices)
            all_best_lists.append(best_list)
            all_prob_lists.append(prob_lists)

        print '\tTime: %f' % (time.time() - start)
        return all_best_lists, all_prob_lists

