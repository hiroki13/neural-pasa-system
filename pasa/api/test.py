from ..utils.io_utils import say, load_data
from ..utils.preprocessor import Preprocessor
from ..model.model_api import ModelAPI, NBestModelAPI


class Tester(object):

    def __init__(self, argv, preprocessor):
        self.argv = argv
        self.config = load_data(argv.load_config)
        preprocessor.argv.window = self.config.window

        self.preprocessor = preprocessor
        self.model_api = None

        self.vocab_word = None
        self.vocab_label = None

        self.corpus_set = None
        self.dev_samples = None
        self.test_samples = None

    def setup_testing(self):
        say('\n\nSETTING UP AN INTRA-SENTENTIAL PASA TESTING SETTING\n')
        self._setup_corpus()
        self._setup_word()
        self._setup_label()
        self._setup_samples()
        self._setup_model_api()

    def _setup_corpus(self):
        self.corpus_set = self._load_corpus_set()
        self._show_corpus_stats(self.corpus_set)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        return self.preprocessor.load_corpus_set()

    def _show_corpus_stats(self, corpus_set):
        self.preprocessor.show_corpus_stats(corpus_set)

    def _setup_word(self):
        self.vocab_word = load_data(self.argv.load_word)

    def _setup_label(self):
        self.vocab_label = load_data(self.argv.load_label)

    def _setup_samples(self):
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]

    def _setup_model_api(self):
        say('\n\nSetting up a model API...\n')
        model_api = self.model_api = ModelAPI(argv=self.config,
                                              emb=None,
                                              vocab_word=self.vocab_word,
                                              vocab_label=self.vocab_label)
        model_api.set_output()
        model_api.set_model()
        model_api.compile_model()
        model_api.set_decoder()
        model_api.load_params(self.argv.load_params)
        model_api.set_predict_f()

    def predict(self):
        argv = self.argv
        model_api = self.model_api

        if argv.dev_data:
            print '\n  DEV\n\t',
            dev_results, dev_results_prob = model_api.predict_all(self.dev_samples)
            dev_f1 = model_api.eval_all(dev_results, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if argv.test_data:
            print '\n  TEST\n\t',
            test_results, test_results_prob = model_api.predict_all(self.test_samples)
            test_f1 = model_api.eval_all(test_results, self.test_samples)
            say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))


class JackKnifeTester(Tester):

    def __init__(self, argv, preprocessor):
        super(JackKnifeTester, self).__init__(argv, preprocessor)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        train_set = load_data(self.argv.train_data)
        train_corpus, test_corpus = self.separate_train_part(train_set, self.argv.sec)
        dev_corpus = self.preprocessor.corpus_loader.load_corpus(self.argv.dev_data)
        return train_corpus, dev_corpus, test_corpus

    @staticmethod
    def separate_train_part(train_set, sec):
        return None, train_set[sec]

    def _setup_model_api(self):
        say('\n\nSetting up a model API...\n')
        model_api = self.model_api = NBestModelAPI(argv=self.config,
                                                   emb=None,
                                                   vocab_word=self.vocab_word,
                                                   vocab_label=self.vocab_label)
        model_api.set_output()
        model_api.set_model()
        model_api.compile_model()
        model_api.set_decoder()
        model_api.load_params(self.argv.load_params)
        model_api.set_predict_f()

    def predict(self):
        argv = self.argv
        model_api = self.model_api

        if self.dev_samples:
            print '\n  DEV\n\t',
            dev_results, dev_results_prob = model_api.predict_all(self.dev_samples)
            dev_f1 = model_api.eval_all(dev_results, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if self.test_samples:
            print '\n  TEST\n\t',
            test_results, test_results_prob = model_api.predict_all(self.test_samples)
            test_f1 = model_api.eval_all(test_results, self.test_samples)
            say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))
            test_n_best_lists = model_api.predict_n_best_lists(self.test_samples)

            output_fn = 'sec%d' % argv.sec
            output_dir = 'data/rerank/list/layers%d/best%d/' % (model_api.argv.layers, model_api.argv.n_best)
            if argv.output_dir is not None:
                output_dir = argv.output_dir
            model_api.save_n_best_lists(output_fn, output_dir, test_n_best_lists)


def select_tester(argv):
    if argv.model == 'base':
        return Tester(argv, Preprocessor(argv))
    elif argv.model == 'jack':
        return JackKnifeTester(argv, Preprocessor(argv))


def main(argv):
    tester = select_tester(argv)
    tester.setup_testing()
    tester.predict()
