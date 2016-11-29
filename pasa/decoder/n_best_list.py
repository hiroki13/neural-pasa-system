from ..experimenter.evaluator import Eval


class NBestList(object):

    def __init__(self, sample, n_best):
        self.words = sample.sent
        self.word_ids = sample.word_ids
        self.label_ids = sample.label_ids
        self.prd_indices = sample.prd_indices
        self.lists = n_best

    def get_max_f1_list_index(self):
        best_list = Eval().select_best_f1_list(self.lists, self.label_ids)
        return self.lists.index(best_list)
