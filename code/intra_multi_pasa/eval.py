import numpy as np

GA_ID = 1
O_ID = 2
NI_ID = 3


class Eval(object):

    def __init__(self):
        self.corrects = np.zeros(3, dtype='float32')
        self.results_sys = np.zeros(3, dtype='float32')
        self.results_gold = np.zeros(3, dtype='float32')

        self.all_corrects = 0.
        self.all_results_sys = 0.
        self.all_results_gold = 0.

        self.precision = np.zeros(3, dtype='float32')
        self.recall = np.zeros(3, dtype='float32')
        self.f1 = np.zeros(3, dtype='float32')

        self.all_precision = 0.
        self.all_recall = 0.
        self.all_f1 = 0.

        self.nll = 0.

    def update_results(self, batch_y_hat, batch_y):
        assert len(batch_y_hat) == len(batch_y)
        assert len(batch_y_hat[0]) == len(batch_y[0])

        for i in xrange(len(batch_y_hat)):
            sent_y_hat = batch_y_hat[i]
            sent_y = batch_y[i]
            for j in xrange(len(sent_y_hat)):
                y_hat = sent_y_hat[j]
                y = sent_y[j]

                """ Check if the predicted labels are correct or not """
                if y_hat == y:
                    if y_hat == GA_ID:
                        self.corrects[0] += 1
                    elif y_hat == O_ID:
                        self.corrects[1] += 1
                    elif y_hat == NI_ID:
                        self.corrects[2] += 1

                """ Set predicted labels """
                if y_hat == GA_ID:
                    self.results_sys[0] += 1
                elif y_hat == O_ID:
                    self.results_sys[1] += 1
                elif y_hat == NI_ID:
                    self.results_sys[2] += 1

                """ Set gold labels """
                if y == GA_ID:
                    self.results_gold[0] += 1
                elif y == O_ID:
                    self.results_gold[1] += 1
                elif y == NI_ID:
                    self.results_gold[2] += 1

    def set_metrics(self):
        for c in xrange(3):
            self.precision[c] = self.corrects[c] / self.results_sys[c]
            self.recall[c] = self.corrects[c] / self.results_gold[c]
            self.f1[c] = 2 * self.precision[c] * self.recall[c] / (self.precision[c] + self.recall[c])

        self.all_corrects = np.sum(self.corrects)
        self.all_results_sys = np.sum(self.results_sys)
        self.all_results_gold = np.sum(self.results_gold)

        self.all_precision = self.all_corrects / self.all_results_sys
        self.all_recall = self.all_corrects / self.all_results_gold
        self.all_f1 = 2 * self.all_precision * self.all_recall / (self.all_precision + self.all_recall)

    def show_results(self):
        self.set_metrics()
        print '\n\tEVALUATION'

        print '\t\tNLL: %f' % self.nll

        for c in xrange(3):
            print '\n\tCASE: %d' % c
            print '\t\tCRR: %d  TTL P: %d  TTL R: %d' % (self.corrects[c], self.results_sys[c], self.results_gold[c])
            print '\t\tF: %f  P: %f  R: %f' % (self.f1[c], self.precision[c], self.recall[c])

        print '\n\tTOTAL'
        print '\t\tCRR: %d  TTL P: %d  TTL R: %d' % (self.all_corrects, self.all_results_sys, self.all_results_gold)
        print '\t\tF: %f  P: %f  R: %f' % (self.all_f1, self.all_precision, self.all_recall)

