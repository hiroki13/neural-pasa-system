import re

n_cases = 3

"""
    An example of the pas_info:
    [u'alt="active"', u'ga="2"', u'ga_type="dep"', u'ni="1"', u'ni_type="dep"', u'type="pred"']
"""


class Word(object):

    def __init__(self, index, elem, file_encoding='utf-8'):
        self.index = index
        self.form = elem[0].decode(file_encoding)
        self.chars = [c for c in self.form]
        self.cpos = elem[3].decode(file_encoding)
        self.pos = elem[4].decode(file_encoding)

        self.pas_info = elem[-1].decode(file_encoding).split('/')
        self.alt = self._set_alt(self.pas_info)

        self.id = self._set_id()
        self.is_prd = self._set_is_prd()
        self.case_arg_ids = self._set_case_arg_ids()  # cases: [Ga, O, Ni]; elem=arg id
        self.case_types = [-1, -1, -1]
        self.case_arg_index = [-1, -1, -1]
        self.inter_case_arg_index = [[], [], []]

        self.sent_index = -1
        self.chunk_index = -1
        self.chunk_head = -1

    def _set_id(self):
        for p in self.pas_info:
            if 'id=' == p[:3]:
                m = re.search("\d+", p)
                return int(m.group())
        return -1

    def _set_is_prd(self):
        if 'type="pred"' in self.pas_info:
            return True
        return False

    @staticmethod
    def _set_alt(pas_info):
        for p in pas_info:
            if 'alt=' == p[:4]:
                return p[5:-1]
        return '_'

    def _set_case_arg_ids(self):
        case_arg_ids = [-1 for i in xrange(n_cases)]  # -1 is no-corresponding arg

        if self.is_prd is False:
            return case_arg_ids

        for info in self.pas_info:
            """ Extract a case label """
            if 'ga=' == info[:3]:
                case_label = 0
            elif 'o=' == info[:2]:
                case_label = 1
            elif 'ni=' == info[:3]:
                case_label = 2
            else:
                continue

            """ Extract a case arg id """
            exo1 = re.search("exo1", info)
            exo2 = re.search("exo2", info)
            exog = re.search("exog", info)

            if exo1 is not None:
                arg_id = 1001
            elif exo2 is not None:
                arg_id = 1002
            elif exog is not None:
                arg_id = 1003
            else:
                anaphora = re.search("\d+", info)
                arg_id = int(anaphora.group())

            """ Add the case label if the arg is a case """
            case_arg_ids[case_label] = arg_id

        return case_arg_ids

    def set_cases(self, sent, doc):
        # 0=bunsetsu, 1=dep, 2=intra-zero, 3=inter-zero, 4=exophora

        if self.is_prd is False:
            return

        """ Intra-sentential arguments """
        for w in sent:
            for case_label, a_id in enumerate(self.case_arg_ids):
                if w.id == a_id > -1:
                    if w.chunk_index == self.chunk_index:
                        case_type = 0  # Within bunsetsu
                    elif w.chunk_index == self.chunk_head or w.chunk_head == self.chunk_index:
                        case_type = 1  # Dep
                        self.case_arg_index[case_label] = w.index
                    else:
                        case_type = 2  # Zero
                        self.case_arg_index[case_label] = w.index
                    self.case_types[case_label] = case_type

        """ Inter-sentential zero arguments """
        for i, prev_sent in enumerate(doc):
            for w in prev_sent:
                for case_label, a_id in enumerate(self.case_arg_ids):
                    if w.id == a_id > -1 and self.case_types[case_label] < 0:
                        self.case_types[case_label] = 3
                        self.inter_case_arg_index[case_label].append((i, w.index))

        """ Exophora arguments """
        for case_label, a_id in enumerate(self.case_arg_ids):
            if 1000 < a_id:
                self.case_types[case_label] = 4

    def has_args(self):
        for arg_index in self.case_arg_index:
            if arg_index > -1:
                return True
        return False
