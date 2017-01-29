import re

from vocab import GA_INDEX, O_INDEX, NI_INDEX

N_CASES = 3
EXO1 = 1001
EXO2 = 1002
EXOG = 1003

BST = 0
DEP = 1
INTRA_ZERO = 2
INTER_ZERO = 3
EXO = 4


"""
    An example of the pas_info:
    [u'alt="active"', u'ga="2"', u'ga_type="dep"', u'ni="1"', u'ni_type="dep"', u'type="pred"']
"""


class Word(object):

    def __init__(self, index, elem, file_encoding='utf-8'):
        self.index = index
        self.form = elem[0].decode(file_encoding)
        self.pas_info = elem[-1].decode(file_encoding).split('/')
        self.alt = self._set_alt(self.pas_info)

        self.id = self._set_id(self.pas_info)
        self.is_prd = self._set_is_prd(self.pas_info)
        self.arg_ids = self._set_arg_ids()
        self.arg_indices = [-1, -1, -1]
        self.arg_types = [-1, -1, -1]

        self.sent_index = -1
        self.chunk_index = -1
        self.chunk_head = -1

    @staticmethod
    def _set_id(pas_info):
        for p in pas_info:
            if 'id=' == p[:3]:
                m = re.search("\d+", p)
                return int(m.group())
        return -1

    @staticmethod
    def _set_is_prd(pas_info):
        if 'type="pred"' in pas_info:
            return True
        return False

    @staticmethod
    def _set_alt(pas_info):
        for p in pas_info:
            if 'alt=' == p[:4]:
                return p[5:-1]
        return '_'

    def _set_arg_ids(self):
        arg_ids = [-1 for i in xrange(N_CASES)]

        if self.is_prd is False:
            return arg_ids

        for info in self.pas_info:
            case_label = self._extract_case_label(info)
            if case_label is None:
                continue
            arg_ids[case_label] = self._extract_arg_id(info)

        return arg_ids

    @staticmethod
    def _extract_case_label(info):
        if 'ga=' == info[:3]:
            case_index = GA_INDEX
        elif 'o=' == info[:2]:
            case_index = O_INDEX
        elif 'ni=' == info[:3]:
            case_index = NI_INDEX
        else:
            case_index = None
        return case_index

    @staticmethod
    def _extract_arg_id(info):
        exo1 = re.search("exo1", info)
        exo2 = re.search("exo2", info)
        exog = re.search("exog", info)

        if exo1 is not None:
            arg_id = EXO1
        elif exo2 is not None:
            arg_id = EXO2
        elif exog is not None:
            arg_id = EXOG
        else:
            anaphora = re.search("\d+", info)
            arg_id = int(anaphora.group())
        return arg_id

    def set_cases(self, sent):
        if self.is_prd is False:
            return
        self._set_intra_cases(sent)

    def _set_intra_cases(self, sent):
        for w in sent:
            for case_index, a_id in enumerate(self.arg_ids):
                if w.id == a_id > -1:
                    if w.chunk_index == self.chunk_index:
                        case_type = BST
                    elif w.chunk_index == self.chunk_head or w.chunk_head == self.chunk_index:
                        case_type = DEP
                        self.arg_indices[case_index] = w.index
                    else:
                        case_type = INTRA_ZERO
                        self.arg_indices[case_index] = w.index

#                    if self.arg_types[case_index] == -1 or self.arg_types[case_index] > case_type:
                    self.arg_types[case_index] = case_type

    def has_args(self):
        for arg_index in self.arg_indices:
            if arg_index > -1:
                return True
        return False


class ConllWord(object):

    def __init__(self, elem, file_encoding='utf-8'):
        self.index = int(elem[0])
        self.form = elem[1].decode(file_encoding)
        self.cpos = elem[2].decode(file_encoding)
        self.pos = elem[3].decode(file_encoding)
        self.alt = elem[4].decode(file_encoding)
        self.chunk_index = int(elem[5])
        self.chunk_head = int(elem[6])
