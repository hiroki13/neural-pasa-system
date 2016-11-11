from io_utils import say
from eval import Eval
from ..ling.n_best_list import NBestList


class NBestListFactory(object):

    def __init__(self, N):
        self.N = N

    @staticmethod
    def create_n_best_lists(results):
        say('\n\n  Create N-best list\n')
        return [NBestList(sample, n_best) for sample, n_best in zip(results.samples, results.decoder_outputs)]

    @staticmethod
    def eval_n_best_list(n_best_lists):
        list_eval = Eval()
        for n_best_list in n_best_lists:
            if len(n_best_list.prd_indices) == 0:
                continue
            best_f1_list = list_eval.select_best_f1_list(n_best_list=n_best_list.lists, batch_y=n_best_list.label_ids)
            list_eval.update_results(batch_y_hat=best_f1_list, batch_y=n_best_list.label_ids)
        list_eval.show_results()
        say('\n\n')

