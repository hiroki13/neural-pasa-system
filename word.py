import re

n_cases = 3


class Word(object):

    def __init__(self, index, elem, file_encoding='utf-8'):
        self.index = index
        self.form = elem[0].decode(file_encoding)
        self.pas_info = elem[-1].decode(file_encoding).split('/')
        """ An example of the pas_info:
            [u'alt="active"', u'ga="2"', u'ga_type="dep"', u'ni="1"', u'ni_type="dep"', u'type="pred"']
        """

        self.pas_id = self.set_pas_id()
        self.is_prd = self.set_is_prd()
        self.case_arg_ids = self.set_case_arg_ids()  # cases: [Ga, O, Ni]; elem=arg id
        self.Ga_type = -1
        self.O_type = -1
        self.Ni_type = -1
        self.set_case_type()

    def set_pas_id(self):
        for p in self.pas_info:
            if 'id=' == p[:3]:
                m = re.search("\d+", p)
                return int(m.group())
        return -1

    def set_is_prd(self):
        if 'type="pred"' in self.pas_info:
            return True
        return False

    def set_case_arg_ids(self):
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

    def set_case_type(self):
        if 'ga_type="dep"' in self.pas_info:
            self.Ga_type = 1
        elif 'ga_type="zero"' in self.pas_info:
            self.Ga_type = 0
        if 'o_type="dep"' in self.pas_info:
            self.O_type = 1
        elif 'o_type="zero"' in self.pas_info:
            self.O_type = 0
        if 'ni_type="dep"' in self.pas_info:
            self.Ni_type = 1
        elif 'ni_type="zero"' in self.pas_info:
            self.Ni_type = 0

