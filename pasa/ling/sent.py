from word import BST, DEP, INTRA_ZERO, INTER_ZERO
from pas import PAS, Argument

GA = 'Ga'
O = 'O'
NI = 'Ni'


class Sentence(object):

    def __init__(self):
        self.words = []
        self.pas = []

    def size(self):
        return len(self.words)

    def size_prds(self):
        return len(self.pas)

    def get_word(self, index):
        return self.words[index]

    def set_prd(self, prd):
        prd_index = int(prd[-1].split(':')[0])
        self.pas.append(PAS(self.get_word(prd_index)))

    def set_args(self, res):
        label_type = res[1]
        info = res[-1].split(' ')

        if len(res) < 3:
            args = []
        else:
            args = [self._gen_arg(res_info) for res_info in info]

        if label_type == 'Gold':
            self.pas[-1].register_gold(args)
        else:
            self.pas[-1].register_sys(args)

    def _gen_arg(self, res_info):
        elems = res_info.split(':')
        case_name = elems[0]
        case_id = self._get_case_id(case_name)
        word_index = int(elems[1])
        prd_index = self.pas[-1].prd_index()
        case_type = self._get_case_type(word_index, prd_index)
        return Argument((word_index, prd_index, case_id, case_type))

    @staticmethod
    def _get_case_id(case_name):
        case_id = -1
        if case_name == GA:
            case_id = 0
        elif case_name == O:
            case_id = 1
        elif case_name == NI:
            case_id = 2
        assert case_id > -1, case_name
        return case_id

    def _get_case_type(self, word_index, prd_index):
        arg = self.get_word(word_index)
        prd = self.get_word(prd_index)
        if arg.chunk_index == prd.chunk_index:
            case_type = BST
        elif arg.chunk_index == prd.chunk_head or arg.chunk_head == prd.chunk_index:
            case_type = DEP
        else:
            case_type = INTRA_ZERO
        return case_type


