class PAS(object):

    def __init__(self, prd):
        self.prd = prd
        self.args_sys = [[], [], []]
        self.args_gold = [[], [], []]

    def prd_index(self):
        return self.prd.index

    def register_sys(self, args):
        for arg in args:
            self.args_sys[arg.case_index].append(arg)

    def register_gold(self, args):
        for arg in args:
            self.args_gold[arg.case_index].append(arg)


class Argument(object):

    def __init__(self, elem):
        self.word_index = elem[0]
        self.prd_index = elem[1]
        self.case_index = elem[2]
        self.case_type = elem[3]
