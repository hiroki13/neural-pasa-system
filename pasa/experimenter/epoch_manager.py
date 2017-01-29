import numpy as np

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

            model_api.train_one_epoch(train_samples)
            dev_results, update, trainable_emb = self._validate(epoch, model_api, dev_samples, untrainable_emb)
            test_results = self._test(epoch, model_api, test_samples, update)

            if argv.save and update:
                model_api.save_model()
#                if test_results:
#                    model_api.save_pas_results(results=test_results, samples=test_samples)
#                    model_api.save_outputs(results=test_results)

            if trainable_emb:
                model_api.model.emb_layer.word_emb.set_value(trainable_emb)

            self._show_results()

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
            f1 = model_api.eval_one_epoch(batch_y_hat=results, samples=samples)
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
            f1 = model_api.eval_one_epoch(batch_y_hat=results, samples=samples)
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
