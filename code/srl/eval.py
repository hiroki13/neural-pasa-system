import numpy as np


class Eval(object):

    def __init__(self, vocab_label):
        self.vocab_label = vocab_label
        self.n_labels = vocab_label.size() - 2

        self.corrects = np.zeros(self.n_labels, dtype='float32')
        self.results_sys = np.zeros(self.n_labels, dtype='float32')
        self.results_gold = np.zeros(self.n_labels, dtype='float32')

        self.all_corrects = 0.
        self.all_results_sys = 0.
        self.all_results_gold = 0.

        self.precision = np.zeros(self.n_labels, dtype='float32')
        self.recall = np.zeros(self.n_labels, dtype='float32')
        self.f1 = np.zeros(self.n_labels, dtype='float32')

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
                if y_hat == y and y_hat > 1:
                    self.corrects[y_hat-2] += 1

                """ Set predicted labels """
                if y_hat > 1:
                    self.results_sys[y_hat-2] += 1

                """ Set gold labels """
                if y > 1:
                    self.results_gold[y-2] += 1

    def set_metrics(self):
        for c in xrange(self.n_labels):
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

#        for c in xrange(self.n_labels):
#            print '\n\tCASE: %d' % c
#            print '\t\tCRR: %d  TTL P: %d  TTL R: %d' % (self.corrects[c], self.results_sys[c], self.results_gold[c])
#            print '\t\tF: %f  P: %f  R: %f' % (self.f1[c], self.precision[c], self.recall[c])

        print '\n\tTOTAL'
        print '\t\tCRR: %d  TTL P: %d  TTL R: %d' % (self.all_corrects, self.all_results_sys, self.all_results_gold)
        print '\t\tF: %f  P: %f  R: %f' % (self.all_f1, self.all_precision, self.all_recall)

