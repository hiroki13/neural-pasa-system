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

            train_samples = self.shuffle_batches(train_samples, 3)
            self._train_one_epoch(model_api, train_samples)
            update, trainable_emb = self._validate(epoch, model_api, dev_samples, untrainable_emb)
            self._test(epoch, model_api, test_samples, update)

            if trainable_emb:
                model_api.model.emb_layer.word_emb.set_value(trainable_emb)

            self._show_results()

    def _train_one_epoch(self, model_api, samples):
        argv = self.argv
        dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
        model_api.model.dropout.set_value(dropout_p)
        model_api.train_each(samples)

    def _validate(self, epoch, model_api, samples, untrainable_emb=None):
        argv = self.argv

        if untrainable_emb:
            trainable_emb = model_api.model.emb_layer.word_emb.get_value(True)
            model_api.model.emb_layer.word_emb.set_value(np.r_[trainable_emb, untrainable_emb])
        else:
            trainable_emb = None

        update = False
        if argv.dev_data:
            print '\n  DEV\n\t',
            results_list, results_prob = model_api.predict_all(samples)
            f1 = model_api.eval_all(results_list, samples)
            if self.best_f1 < f1:
                self.best_f1 = f1
                self.f1_history[epoch+1] = [f1]
                update = True

        return update, trainable_emb

    def _test(self, epoch, model_api, samples, update):
        argv = self.argv

        if argv.test_data:
            print '\n  TEST\n\t',
            results_list, results_prob = model_api.predict_all(samples)
            f1 = model_api.eval_all(results_list, samples)
            if update:
                if epoch + 1 in self.f1_history:
                    self.f1_history[epoch+1].append(f1)
                else:
                    self.f1_history[epoch+1] = [f1]

    def _show_results(self):
        say('\n\n\tF1 HISTORY')
        for k, v in sorted(self.f1_history.items()):
            if len(v) == 2:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
            else:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
        say('\n\n')

    def shuffle_batches(self, batches, n_inputs):
        """
        :param batches: 1D: n_sents; Sample
        """
        new_batches = []
        batch = [[] for i in xrange(n_inputs)]

        batches = self.separate_batches(batches)
        batches = self._sort_by_n_words(batches)
        prev_n_words = len(batches[0][0])

        for sample in batches:
            n_words = len(sample[0])
            boundary_elems = (n_words, prev_n_words, len(batch[-1]))

            if self._is_batch_boundary(boundary_elems, self.argv.batch_size):
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
    def _is_batch_boundary(boundary_elems, batch_size):
        n_words, prev_n_words, n_batches = boundary_elems
        if prev_n_words != n_words or n_batches >= batch_size:
            return True
        return False

    @staticmethod
    def _add_inputs_to_batch(batch, sample):
        batch[0].append(sample[0])
        batch[1].append(sample[1])
        batch[2].append(sample[2])
        return batch

    @staticmethod
    def _sort_by_n_words(samples):
        np.random.shuffle(samples)
        samples.sort(key=lambda s: len(s[0]))
        return samples
