import os
import gzip
import cPickle as pickle

from ..utils.io_utils import move_data


class IOManager(object):

    def __init__(self, argv, vocab_word, vocab_label):
        self.argv = argv
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label
        self.output_fn = self._set_output_fn(argv)
        self.output_dir = self._set_output_dir(argv)

    @staticmethod
    def _set_output_fn(argv):
        if argv.output_fn is not None:
            return argv.output_fn

        output_fn = 'model-%s.layers-%d.batch-%d.reg-%f' % (argv.model,
                                                            argv.layers,
                                                            argv.batch_size,
                                                            argv.reg)
        if argv.model == 'jack':
            if argv.sec is None:
                output_fn += '.all'
            else:
                output_fn += '.sec-%d' % argv.sec
        return output_fn

    def _set_output_dir(self, argv):
        if argv.output_dir is not None:
            output_dir = argv.output_dir
        else:
            output_dir = 'data/%s/' % argv.model
        self._create_path(output_dir)
        return output_dir

    @staticmethod
    def _create_path(output_path):
        path = ''
        dir_names = output_path.split('/')
        for dir_name in dir_names:
            path += dir_name
            if not os.path.exists(path):
                os.mkdir(path)
            path += '/'

    @staticmethod
    def _check_identifier(fn):
        if not fn.endswith(".pkl.gz"):
            fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
        return fn

    def save_model(self, model):
        self._save_params(model, self.output_fn, self.output_dir)
        self._save_config(self.output_fn, self.output_dir)

    @staticmethod
    def load_params(model, path):
        with gzip.open(path) as fin:
            params = pickle.load(fin)
            assert len(model.layers) == len(params)
            for l, p in zip(model.layers, params):
                for p1, p2 in zip(l.params, p):
                    p1.set_value(p2.get_value(borrow=True))
        return model

    def _save_params(self, model, fn, output_dir):
        fn = 'param.' + fn
        fn = self._check_identifier(fn)
        with gzip.open(fn, "w") as fout:
            pickle.dump([l.params for l in model.layers], fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        output_dir += 'param'
        move_data(fn, output_dir)

    def _save_config(self, fn, output_dir):
        fn = 'config.' + fn
        fn = self._check_identifier(fn)
        with gzip.open(fn, "w") as fout:
            pickle.dump(self.argv, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        output_dir += 'config'
        move_data(fn, output_dir)

    def save_outputs(self, results):
        self._save_results(results)

    def _save_results(self, results):
        fn = 'result.' + self.output_fn
        fn = self._check_identifier(fn)
        with gzip.open(fn, "w") as fout:
            pickle.dump(results, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)
        output_dir = self.output_dir + 'result'
        move_data(fn, output_dir)

    def save_pas_results(self, results, samples):
        fn = 'pas.' + self.output_fn + '.txt'
        self._output_analyzed_pas(fn, results, samples)
        output_dir = self.output_dir + 'pas'
        move_data(fn, output_dir)

    def _output_analyzed_pas(self, fn, results, samples):
        assert len(results) == len(samples)
        with open(fn, 'w') as fout:
            for result, sample in zip(results, samples):
                text = self._generate_sent_info(sample.sent)
                fout.writelines(text.encode('utf-8'))
                text = self._generate_analyzed_pas_info(result, sample)
                fout.writelines(text.encode('utf-8'))

    def _generate_analyzed_pas_info(self, result_sys, sample):
        sent = sample.sent
        prds = [sent[prd_index] for prd_index in sample.prd_indices]
        assert len(result_sys) == len(prds)

        text = ''
        for prd_i, (r_s, prd) in enumerate(zip(result_sys, prds)):
            prd_index = sample.prd_indices[prd_i]
            prd = sent[prd_index]
            text += '#\tPRD\t%d:%s\n' % (prd_index, prd.form)
            text += '*\tGold\t'
            text += self._generate_analyzed_pas_info_gold(sent, prd)
            text += '\n*\tSys\t'
            text += self._generate_analyzed_pas_info_sys(sent, r_s)
            text += '\n'
        text += '\n'
        return text

    def _generate_analyzed_pas_info_gold(self, sent, prd):
        text = ''
        for case_index, index in enumerate(prd.arg_indices):
            if index > -1:
                word = sent[index]
                text += '%s:%d:%s ' % (self.vocab_label.get_word(case_index+1), word.index, word.form)
        return text

    def _generate_analyzed_pas_info_sys(self, sent, labels):
        text = ''
        for word, label in zip(sent, labels):
            if 0 < label < 4:
                text += '%s:%d:%s ' % (self.vocab_label.get_word(label), word.index, word.form)
        return text

    def _generate_sent_info(self, sent):
        text = ''
        for word in sent:
            for info in self._generate_word_info(word):
                if type(info) == int:
                    text += '%d\t' % info
                else:
                    text += '%s\t' % info
            text += '\n'
        return text

    @staticmethod
    def _generate_word_info(word):
        return word.index, word.form, word.chunk_index, word.chunk_head

    def save_stats_test_format(self, results, samples):
        fn = 'stats.' + self.output_fn + '.txt'
        self._output_stats_format(fn, results, samples)
        output_dir = self.output_dir + 'stats'
        move_data(fn, output_dir)

    def _output_stats_format(self, fn, results, samples):
        assert len(results) == len(samples)
        with open(fn, 'w') as fout:
            for result, sample in zip(results, samples):
                sent = sample.sent
                for prd_result, prd_answer, prd_index in zip(result, sample.y, sample.prd_indices):
                    prd = sent[prd_index]
                    for word_index, (case_index1, case_index2) in enumerate(zip(prd_result, prd_answer)):
                        word = sent[word_index]
                        if word.chunk_head == prd.chunk_index or word.chunk_index == prd.chunk_head:
                            arg_type = 'dep'
                        else:
                            arg_type = 'inner'
                        case_name1 = self._get_case_name(case_index1)
                        case_name2 = self._get_case_name(case_index2)
                        text = '%s %s %s\n' % (case_name1, arg_type, case_name2)
                        fout.writelines(text.encode('utf-8'))

    @staticmethod
    def _get_case_name(case_index):
        case_name = 'NONE'
        if case_index == 1:
            case_name = 'ga'
        elif case_index == 2:
            case_name = 'o'
        elif case_index == 3:
            case_name = 'ni'
        return case_name


