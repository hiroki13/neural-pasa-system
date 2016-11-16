import math
import numpy as np

from ..utils.io_utils import say

GA_ID = 1
O_ID = 2
NI_ID = 3


class Eval(object):

    def __init__(self):
        self.corrects = None
        self.results_sys = None
        self.results_gold = None

        self.precision = None
        self.recall = None
        self.f1 = None

        self.all_corrects = None
        self.all_results_sys = None
        self.all_results_gold = None

        self.all_precision = 0.
        self.all_recall = 0.
        self.all_f1 = 0.
        self.nll = 0.

        self._set_params()

    def _set_params(self):
        self.corrects = np.zeros((3, 3), dtype='float32')
        self.results_sys = np.zeros((3, 3), dtype='float32')
        self.results_gold = np.zeros((3, 3), dtype='float32')

        self.precision = np.zeros((3, 3), dtype='float32')
        self.recall = np.zeros((3, 3), dtype='float32')
        self.f1 = np.zeros((3, 3), dtype='float32')

        self.all_corrects = np.zeros(3, dtype='float32')
        self.all_results_sys = np.zeros(3, dtype='float32')
        self.all_results_gold = np.zeros(3, dtype='float32')

    def _summarize(self):
        self.precision = self.corrects / self.results_sys
        self.recall = self.corrects / self.results_gold
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        self.all_corrects = np.sum(self.corrects[:, 1:])
        self.all_results_sys = np.sum(self.results_sys[:, 1:])
        self.all_results_gold = np.sum(self.results_gold[:, 1:])

        self.all_precision = self.all_corrects / self.all_results_sys
        self.all_recall = self.all_corrects / self.all_results_gold
        self.all_f1 = 2 * self.all_precision * self.all_recall / (self.all_precision + self.all_recall)

    def show_results(self):
        self._summarize()
        say('\n\tNLL: %f' % self.nll)
        say('\n\n\tACCURACY')

        corrects = np.sum(self.corrects[:, 1:], axis=1)
        results_sys = np.sum(self.results_sys[:, 1:], axis=1)
        results_gold = np.sum(self.results_gold[:, 1:], axis=1)
        precision = corrects / results_sys
        recall = corrects / results_gold
        f1 = 2 * precision * recall / (precision + recall)

        for case_index, (correct, result_sys, result_gold) in enumerate(zip(self.corrects,
                                                                            self.results_sys,
                                                                            self.results_gold)):
            if case_index == 0:
                case_name = 'GA'
            elif case_index == 1:
                case_name = 'O'
            elif case_index == 2:
                case_name = 'Ni'
            else:
                say('\nERROR\n')
                exit()

            say('\n\tCASE-%s:\n' % case_name)
            say('\tALL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
                f1[case_index], precision[case_index], int(corrects[case_index]), int(results_sys[case_index]),
                recall[case_index], int(corrects[case_index]), int(results_gold[case_index])))

            for case_type, (crr, r_sys, r_gold) in enumerate(zip(correct,
                                                                 result_sys,
                                                                 result_gold)):
                if case_type == 0:
                    continue
                elif case_type == 1:
                    case = 'DEP'
                elif case_type == 2:
                    case = 'ZERO'

                crr = int(crr)
                r_sys = int(r_sys)
                r_gold = int(r_gold)
                say('\t{}:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})\n'.format(
                    case,
                    self.f1[case_index][case_type],
                    self.precision[case_index][case_type], crr, r_sys,
                    self.recall[case_index][case_type], crr, r_gold)
                    )

        crr = int(self.all_corrects)
        r_sys = int(self.all_results_sys)
        r_gold = int(self.all_results_gold)

        say('\n\tTOTAL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            self.all_f1, self.all_precision, crr, r_sys, self.all_recall, crr, r_gold)
            )

    def update_results(self, y_hat_batch, sample):
        assert len(y_hat_batch) == len(sample.y)
        for prd_i, (y_hat_prd, y_prd) in enumerate(zip(y_hat_batch, sample.y)):
            assert len(y_hat_prd) == len(y_prd)

            prd_index = sample.prd_indices[prd_i]
            case_types = self._get_case_types_gold(prd_index, sample)
            self.update_results_gold(case_types)

            for word_index, (y_hat, y) in enumerate(zip(y_hat_prd, y_prd)):
                case_index = self._get_case_index(y_hat)
                case_type = self._get_case_type_sys(word_index, prd_index, sample)
                answer = self._judge_answer(y_hat, y)

                if -1 < case_index:
                    self.results_sys[case_index][case_type] += 1
                    self.corrects[case_index][case_type] += answer

    def update_results_gold(self, case_type):
        for i, c in enumerate(case_type):
            if -1 < c < 3:
                self.results_gold[i][c] += 1

    @staticmethod
    def _judge_answer(y_hat, y):
        if y_hat == y:
            return 1
        return 0

    @staticmethod
    def _get_case_index(y):
        case = -1
        if y == GA_ID:
            case = 0
        elif y == O_ID:
            case = 1
        elif y == NI_ID:
            case = 2
        return case

    @staticmethod
    def _get_case_types_gold(p_index, sample):
        return sample.sent[p_index].case_types

    def _get_case_type_sys(self, w_index, p_index, sample):
        word = sample.sent[w_index]
        prd = sample.sent[p_index]
        return self._get_case_type(word, prd)

    @staticmethod
    def _get_case_type(word, prd):
        if word.chunk_index == prd.chunk_index:
            case_type = 0  # Within bunsetsu
        elif word.chunk_index == prd.chunk_head or word.chunk_head == prd.chunk_index:
            case_type = 1  # Dep
        else:
            case_type = 2  # Zero
        return case_type

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

                    case_y_hat = self._get_case_index(y_hat)
                    case_y = self._get_case_index(y)
                    results_sys[case_y_hat] += 1
                    results_gold[case_y] += 1

                    if 0 < y < 4 and y_hat == y:
                        corrects[case_y_hat] += 1

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


class TrainEval(Eval):

    def __init__(self):
        super(TrainEval, self).__init__()

    def _set_params(self):
        self.corrects = np.zeros(3, dtype='float32')
        self.results_sys = np.zeros(3, dtype='float32')
        self.results_gold = np.zeros(3, dtype='float32')

        self.precision = np.zeros(3, dtype='float32')
        self.recall = np.zeros(3, dtype='float32')
        self.f1 = np.zeros(3, dtype='float32')

        self.all_corrects = 0.
        self.all_results_sys = 0.
        self.all_results_gold = 0.

    def _summarize(self):
        self.precision = self.corrects / self.results_sys
        self.recall = self.corrects / self.results_gold
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        self.all_corrects = np.sum(self.corrects)
        self.all_results_sys = np.sum(self.results_sys)
        self.all_results_gold = np.sum(self.results_gold)

        self.all_precision = self.all_corrects / self.all_results_sys
        self.all_recall = self.all_corrects / self.all_results_gold
        self.all_f1 = 2 * self.all_precision * self.all_recall / (self.all_precision + self.all_recall)

    def show_results(self):
        self._summarize()
        say('\n\tNLL: %f' % self.nll)
        say('\n\n\tACCURACY')

        for c in xrange(3):
            if c == 0:
                case_name = 'GA'
            elif c == 1:
                case_name = 'O'
            elif c == 2:
                case_name = 'Ni'

            crr = int(self.corrects[c])
            r_sys = int(self.results_sys[c])
            r_gold = int(self.results_gold[c])

            say('\n\t{}:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
                case_name, self.f1[c], self.precision[c], crr, r_sys, self.recall[c], crr, r_gold)
                )

        crr = int(self.all_corrects)
        r_sys = int(self.all_results_sys)
        r_gold = int(self.all_results_gold)

        say('\n\tTOTAL:\tF:{:>7.2%}  P:{:>7.2%} ({:>5}/{:>5})  R:{:>7.2%} ({:>5}/{:>5})'.format(
            self.all_f1, self.all_precision, crr, r_sys, self.all_recall, crr, r_gold)
            )

    def update_results(self, batch_y_hat, batch_y):
        assert len(batch_y_hat) == len(batch_y)
        assert len(batch_y_hat[0]) == len(batch_y[0]), '%s\n%s' % (str(batch_y_hat), str(batch_y))

        for sent_y_hat, sent_y in zip(batch_y_hat, batch_y):
            for y_hat, y in zip(sent_y_hat, sent_y):
                case_y_hat = self._get_case_index(y_hat)
                case_y = self._get_case_index(y)

                if -1 < case_y_hat:
                    self.results_sys[case_y_hat] += 1
                    self.corrects[case_y_hat] += self._judge_answer(y_hat, y)
                if -1 < case_y:
                    self.results_gold[case_y] += 1

    def eval_n_best_list(self, n_best_lists, samples):
        for n_best_list, sample in zip(n_best_lists, samples):
            if len(n_best_list) == 0:
                continue
            best_f1_list = self.select_best_f1_list(n_best_list=n_best_list, batch_y=sample.label_ids)
            self.update_results(batch_y_hat=best_f1_list, batch_y=sample.label_ids)
        self.show_results()
        say('\n\n')
