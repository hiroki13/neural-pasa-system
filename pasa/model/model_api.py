import math
import sys
import time

import theano
import theano.tensor as T

from io_manager import IOManager
from model import Model, GridModel, MentionPairModel
from result import Results
from ..decoder.decoder import Decoder
from ..experimenter.evaluator import Eval, TrainEval
from ..utils.io_utils import say


class ModelAPI(object):

    def __init__(self, argv):
        self.argv = argv
        self.emb = None
        self.vocab_word = None
        self.vocab_label = None

        self.model = None
        self.decoder = None
        self.io_manager = None

        self.train = None
        self.predict = None

    def compile(self, vocab_word, vocab_label, init_emb=None):
        say('\n\nBuilding a model API...\n')
        self.emb = init_emb
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.set_model()
        self.set_decoder()
        self.set_io_manager()

    def set_model(self):
        self.model = Model(argv=self.argv,
                           emb=self.emb,
                           n_vocab=self.vocab_word.size(),
                           n_labels=self.vocab_label.size())
        self.compile_model()

    def compile_model(self):
        self.model.compile(self._get_input_tensor_variables())

    @staticmethod
    def _get_input_tensor_variables():
        # x_w: 1D: batch, 2D: n_words, 3D: 5 + window; word id
        # x_p: 1D: batch, 2D: n_words; posit id
        # y: 1D: batch, 2D: n_words; label id
        return T.itensor3('x_w'), T.imatrix('x_p'), T.imatrix('y')

    def set_decoder(self):
        self.decoder = Decoder(self.argv)

    def set_io_manager(self):
        self.io_manager = IOManager(self.argv, self.vocab_word, self.vocab_label)

    def set_train_f(self):
        model = self.model
        self.train = theano.function(inputs=model.inputs,
                                     outputs=[model.y_pred, model.y_gold, model.nll],
                                     updates=model.update
                                     )

    def set_predict_f(self):
        model = self.model
        outputs = self._select_outputs(self.argv, model)
        self.predict = theano.function(inputs=model.x,
                                       outputs=outputs,
                                       )

    @staticmethod
    def _select_outputs(argv, model):
        outputs = [model.y_prob]
        if argv.output == 'pretrain':
            outputs.append(model.hidden_reps)
        return outputs

    def train_one_epoch(self, batch):
        train_eval = TrainEval()
        start = time.time()
        batch.shuffle_batches()

        for index, one_batch in enumerate(batch.batches):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            result_sys, result_gold, nll = self.train(*one_batch)
            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.nll /= float(len(batch.batches))
        train_eval.show_results()

    def predict_one_epoch(self, samples):
        results = Results(self.argv)
        start = time.time()

        for index, sample in enumerate(samples):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            if sample.n_prds == 0:
                model_outputs = []
                decoder_outputs = []
            else:
                model_outputs = self.predict(*self.create_input_variables(sample))
                decoder_outputs = self.decode(prob_lists=model_outputs[0], prd_indices=sample.prd_indices)

            results.add([sample, model_outputs, decoder_outputs])

        print '\tTime: %f' % (time.time() - start)
        return results

    @staticmethod
    def create_input_variables(sample):
        return sample.x

    def decode(self, prob_lists, prd_indices):
        assert len(prob_lists) == len(prd_indices)
        return self.decoder.decode(prob_lists, prd_indices)

    @staticmethod
    def eval_one_epoch(batch_y_hat, samples):
        pred_eval = Eval()
        assert len(batch_y_hat) == len(samples)
        for result, sample in zip(batch_y_hat, samples):
            if len(result) == 0:
                continue
            pred_eval.update_results(y_hat_batch=result, sample=sample)
        pred_eval.show_results()
        return pred_eval.all_f1

    def save_model(self):
        self.io_manager.save_model(self.model)

    def save_pas_results(self, results, samples):
        self.io_manager.save_pas_results(results, samples)

    def save_outputs(self, results):
        self.io_manager.save_outputs(results)

    def load_params(self, fn):
        self.model = self.io_manager.load_params(self.model, fn)


class GridModelAPI(ModelAPI):

    def set_model(self):
        self.model = GridModel(argv=self.argv,
                               emb=self.emb,
                               n_vocab=self.vocab_word.size(),
                               n_labels=self.vocab_label.size())
        self.compile_model()

    @staticmethod
    def _get_input_tensor_variables():
        # x_w: 1D: batch, 2D: n_prds, 3D: n_words, 4D: 5 + window; elem=word id
        # x_p: 1D: batch, 2D: n_prds, 3D: n_words; elem=posit id
        # y: 1D: batch, 2D: n_prds, 3D: n_words; elem=label id
        return T.itensor4('x_w'), T.itensor3('x_p'), T.itensor3('y')

    @staticmethod
    def create_input_variables(sample):
        inputs = []
        for x in sample.x:
            inputs.append([x])
        return inputs


class MentionPairModelAPI(ModelAPI):

    def set_model(self):
        self.model = MentionPairModel(argv=self.argv,
                                      emb=self.emb,
                                      n_vocab=self.vocab_word.size(),
                                      n_labels=self.vocab_label.size())
        self.compile_model()

    @staticmethod
    def _get_input_tensor_variables():
        # x_w: 1D: batch, 2D: n_phi; word id
        # y: 1D: batch; label id
        return T.imatrix('x_w'), T.ivector('y')

    @staticmethod
    def create_input_variables(sample):
        inputs = []
        for x in sample.x:
            inputs.append([x])
        return inputs

    def train_one_epoch(self, batch):
        crr = 0.
        ttl_p = 0.
        ttl_r = 0.
        ttl_nll = 0.
        start = time.time()
        batch.shuffle_batches()

        for index, one_batch in enumerate(batch.batches):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            result_sys, result_gold, nll = self.train(*one_batch)
            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            for s, g in zip(result_sys, result_gold):
                if s == g == 1:
                    crr += 1
                if s == 1:
                    ttl_p += 1
                if g == 1:
                    ttl_r += 1
            ttl_nll += nll

        precision = crr / ttl_p
        recall = crr / ttl_r
        f1 = 2 * precision * recall / (precision + recall)
        print '\tTime: %f' % (time.time() - start)
        say('\tNLL: %f  F1: %f  Precision: %f (%d/%d)  Recall: %f (%d/%d)' % (ttl_nll, f1, precision, crr, ttl_p,
                                                                              recall, crr, ttl_r))

