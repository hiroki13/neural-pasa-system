import math
import numpy as np

from ..utils.io_utils import say

GA_ID = 1
O_ID = 2
NI_ID = 3


class Eval(object):

    def __init__(self):
        self.corrects = np.zeros(3, dtype='float32')
        self.results_sys = np.zeros(3, dtype='float32')
        self.results_gold = np.zeros(3, dtype='float32')

        self.correct = 0.
        self.total = 0.

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
        assert len(batch_y_hat[0]) == len(batch_y[0]), '%s\n%s' % (str(batch_y_hat), str(batch_y))

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

    def update_rank_results(self, batch_y_hat, batch_y, null_arg_index):
        assert len(batch_y_hat) == len(batch_y)
        assert len(batch_y_hat[0]) == len(batch_y[0])

        for i in xrange(len(batch_y_hat)):
            sent_y_hat = batch_y_hat[i]
            sent_y = batch_y[i]
            for case_index in xrange(len(sent_y_hat)-1):
                y_hat = sent_y_hat[case_index]
                y = sent_y[case_index]

                if y != null_arg_index and y_hat == y:
                    self.corrects[case_index] += 1

                if y_hat != null_arg_index:
                    self.results_sys[case_index] += 1

                if y != null_arg_index:
                    self.results_gold[case_index] += 1

    def select_best_f1_list(self, n_best_list, batch_y):
        best_list = None
        best_f1 = -100.0
        assert len(n_best_list) != 0
        for n_th_list in n_best_list:
            corrects = [0.0 for i in xrange(3)]
            results_sys = [0.0 for i in xrange(3)]
            results_gold = [0.0 for i in xrange(3)]

            for i in xrange(len(n_th_list)):
                sent_y_hat = n_th_list[i]
                sent_y = batch_y[i]
                for j in xrange(len(sent_y_hat)):
                    y_hat = sent_y_hat[j]
                    y = sent_y[j]

                    if y_hat == y:
                        if y_hat == GA_ID:
                            corrects[0] += 1
                        elif y_hat == O_ID:
                            corrects[1] += 1
                        elif y_hat == NI_ID:
                            corrects[2] += 1

                    if y_hat == GA_ID:
                        results_sys[0] += 1
                    elif y_hat == O_ID:
                        results_sys[1] += 1
                    elif y_hat == NI_ID:
                        results_sys[2] += 1

                    if y == GA_ID:
                        results_gold[0] += 1
                    elif y == O_ID:
                        results_gold[1] += 1
                    elif y == NI_ID:
                        results_gold[2] += 1

            f1 = self.calc_f1(corrects, results_sys, results_gold)
            if best_f1 < f1:
                best_f1 = f1
                best_list = n_th_list

        assert best_list is not None
        return best_list

    @staticmethod
    def calc_f1(corrects, results_sys, results_gold):
        all_corrects = np.sum(corrects)
        all_results_sys = np.sum(results_sys)
        all_results_gold = np.sum(results_gold)

        all_precision = all_corrects / all_results_sys
        all_recall = all_corrects / all_results_gold
        all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall)

        if math.isnan(all_f1):
            all_f1 = 0.0

        return all_f1

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
        say('\n\tNLL: %f' % self.nll)

        for c in xrange(3):
            crr = int(self.corrects[c])
            r_sys = int(self.results_sys[c])
            r_gold = int(self.results_gold[c])

            say('\n\tCASE-{:d}:\tF:{:.2%}  P:{:.2%} ({:d}/{:d})  R:{:.2%} ({:d}/{:d})'.format(
                c, self.f1[c], self.precision[c], crr, r_sys, self.recall[c], crr, r_gold)
                )

        crr = int(self.all_corrects)
        r_sys = int(self.all_results_sys)
        r_gold = int(self.all_results_gold)

        say('\n\tTOTAL:\tF:{:.2%}  P:{:.2%} ({:d}/{:d})  R:{:.2%} ({:d}/{:d})'.format(
            self.all_f1, self.all_precision, crr, r_sys, self.all_recall, crr, r_gold)
            )

    def update_rerank_results(self, batch_y_hat, batch_y):
        assert len(batch_y_hat) == len(batch_y)
        for y_hat, y in zip(batch_y_hat, batch_y):
            if y_hat == y:
                self.correct += 1
            self.total += 1

    def show_accuracy(self):
        say('\n\tACCURACY: {:.2%} ({:d}/{:d})'.format(self.correct / self.total, int(self.correct), int(self.total)))
