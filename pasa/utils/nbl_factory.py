from io_utils import say
from eval import Eval
from ..ling.n_best_list import NBestList


class NBestListFactory(object):

    def __init__(self, N):
        self.N = N

    def create_n_best_lists(self, decoder, samples, all_prob_lists):
        say('\n\n  Create N-best list\n')
        all_prd_indices = self._extract_prd_indices(samples)
        assert len(samples) == len(all_prob_lists) == len(all_prd_indices)
        n_bests = self._decode_n_best(decoder, all_prob_lists, all_prd_indices)
        return self._gen_n_best_lists(samples, n_bests)

    def _decode_n_best(self, decoder, all_prob_lists, all_prd_indices):
        return decoder.decode_n_best(all_prob_lists=all_prob_lists, all_prd_indices=all_prd_indices, N=self.N)

    @staticmethod
    def _gen_n_best_lists(samples, n_bests):
        assert len(samples) == len(n_bests), '%d %d' % (len(samples), len(n_bests))
        return [NBestList(sample, n_best) for sample, n_best in zip(samples, n_bests)]

    def eval_n_best_list(self, samples, n_best_lists):
        list_eval = Eval()

        gold_labels = self._extract_label_ids(samples)
        assert len(n_best_lists) == len(gold_labels)

        for n_best_list, batch_y in zip(n_best_lists, gold_labels):
            if len(batch_y) == 0:
                continue
            best_f1_list = list_eval.select_best_f1_list(n_best_list=n_best_list.lists, batch_y=batch_y)
            list_eval.update_results(batch_y_hat=best_f1_list, batch_y=batch_y)
        list_eval.show_results()
        say('\n\n')

    @staticmethod
    def _extract_prd_indices(samples):
        return [sample.prd_indices for sample in samples]

    @staticmethod
    def _extract_label_ids(samples):
        return [sample.label_ids for sample in samples]

