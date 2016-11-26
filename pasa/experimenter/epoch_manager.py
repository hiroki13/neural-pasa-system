import numpy as np
import theano

from ..utils.io_utils import say


class EpochManager(object):

    def __init__(self, argv):
        self.argv = argv
        self.f1_history = {}
        self.best_f1 = -1.

    def train(self, model_api, train_samples, dev_samples, test_samples, untrainable_emb=None):
        argv = self.argv

        for epoch in xrange(argv.epoch):
            say('\nEpoch: %d\n' % (epoch + 1))
            print '  TRAIN\n\t',

            train_samples = self.shuffle_batches(train_samples)

            self._train_one_epoch(model_api, train_samples)
            dev_results, update, trainable_emb = self._validate(epoch, model_api, dev_samples, untrainable_emb)
            test_results = self._test(epoch, model_api, test_samples, update)

            if argv.save and update:
                model_api.save_model()
                if test_results:
                    model_api.save_pas_results(results=test_results.decoder_outputs, samples=test_samples)
                    model_api.save_outputs(results=test_results)

            if trainable_emb:
                model_api.model.emb_layer.word_emb.set_value(trainable_emb)

            self._show_results()

    def _train_one_epoch(self, model_api, samples):
        argv = self.argv
        dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
        model_api.model.dropout.set_value(dropout_p)
        model_api.train_one_epoch(samples)

    def _validate(self, epoch, model_api, samples, untrainable_emb=None):
        results = None

        if untrainable_emb:
            trainable_emb = model_api.model.emb_layer.word_emb.get_value(True)
            model_api.model.emb_layer.word_emb.set_value(np.r_[trainable_emb, untrainable_emb])
        else:
            trainable_emb = None

        update = False
        if samples:
            print '\n  DEV\n\t',
            # results: (sample, result, decoded_result)
            results = model_api.predict_one_epoch(samples)
            f1 = model_api.eval_one_epoch(batch_y_hat=results.decoder_outputs, samples=samples)
            if self.best_f1 < f1:
                self.best_f1 = f1
                self.f1_history[epoch+1] = [f1]
                update = True

        return results, update, trainable_emb

    def _test(self, epoch, model_api, samples, update):
        results = None

        if samples:
            print '\n  TEST\n\t',
            results = model_api.predict_one_epoch(samples)
            f1 = model_api.eval_one_epoch(batch_y_hat=results.decoder_outputs, samples=samples)
            if update:
                if epoch + 1 in self.f1_history:
                    self.f1_history[epoch+1].append(f1)
                else:
                    self.f1_history[epoch+1] = [f1]

        return results

    def _show_results(self):
        say('\n\n\tF1 HISTORY')
        for k, v in sorted(self.f1_history.items()):
            if len(v) == 2:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
            else:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
        say('\n\n')

    def shuffle_batches(self, batches):
        new_batches = []
        n_inputs = len(batches[0])
        batch = [[] for i in xrange(n_inputs)]

        batches = self.separate_batches(batches)
        batches = self._sort_by_n_words(batches)
        prev_n_words = len(batches[0][0])

        for sample in batches:
            n_words = len(sample[0])
            elems = (n_words, prev_n_words, len(batch[-1]))

            if self._is_batch_boundary(elems, self.argv.batch_size):
                prev_n_words = n_words
                new_batches.append(batch)
                batch = [[] for i in xrange(n_inputs)]
            batch = self._add_inputs_to_batch(batch, sample)

        if len(batch[0]) > 0:
            new_batches.append(batch)

        return new_batches

    @staticmethod
    def separate_batches(batches):
        return [elem for batch in batches for elem in zip(*batch)]

    @staticmethod
    def _is_batch_boundary(elems, batch_size):
        n_words, prev_n_words, n_batches = elems
        if prev_n_words != n_words or n_batches >= batch_size:
            return True
        return False

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: len(s[0]))
        return samples

    @staticmethod
    def _add_inputs_to_batch(batch, sample):
        batch[0].append(sample[0])
        batch[1].append(sample[1])
        batch[2].append(sample[2])
        return batch


class JointEpochManager(EpochManager):

    def __init__(self, argv):
        super(JointEpochManager, self).__init__(argv)

    def shuffle_batches(self, batches):
        new_batches = []
        n_inputs = len(batches[0])
        batch = [[] for i in xrange(n_inputs)]

        samples = self.separate_batches(batches)
        samples = self._sort_by_n_words(samples)
        prev_n_prds = len(samples[0][2])
        prev_n_words = len(samples[0][2][0])

        for sample in samples:
            n_prds = len(sample[2])
            n_words = len(sample[2][0])
            elems = (n_words, prev_n_words, n_prds, prev_n_prds, len(batch[2]))

            if self._is_batch_boundary(elems, self.argv.batch_size):
                prev_n_prds = n_prds
                prev_n_words = n_words
                new_batches.append(batch)
                batch = [[] for i in xrange(n_inputs)]

            batch = self._add_inputs_to_batch(batch, sample)

        if len(batch[0]) > 0:
            new_batches.append(batch)

        return new_batches

    @staticmethod
    def _is_batch_boundary(elems, batch_size):
        n_words, prev_n_words, n_prds, prev_n_prds, n_batches = elems
        if prev_n_words != n_words or n_prds != prev_n_prds or n_batches >= batch_size:
            return True
        return False

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: len(s[2]))
        samples.sort(key=lambda s: len(s[2][0]))
        return samples

