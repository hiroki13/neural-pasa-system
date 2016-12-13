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
                text = self._generate_analyzed_pas_info(result, sample.label_ids, sample.prd_indices, sample.sent)
                fout.writelines(text.encode('utf-8'))

    def _generate_analyzed_pas_info(self, result_sys, result_gold, prd_indices, sent):
        text = ''
        for r_s, r_g, prd_index in zip(result_sys, result_gold, prd_indices):
            prd = sent[prd_index]
            text += '#\tPRD\t%d:%s:%s:%s\n' % (prd_index, prd.form, prd.cpos, prd.alt)
            text += '*\tGold\t'
            text += self._generate_analyzed_pas_info_each(sent, r_g)
            text += '\n*\tSys\t'
            text += self._generate_analyzed_pas_info_each(sent, r_s)
            text += '\n'
        text += '\n'
        return text

    def _generate_analyzed_pas_info_each(self, sent, labels):
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
        return word.index, word.form, word.cpos, word.pos, word.alt, word.chunk_index, word.chunk_head

