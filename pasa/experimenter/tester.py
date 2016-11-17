from experimenter import Experimenter

from ..utils.io_utils import say


class Tester(Experimenter):

    def __init__(self, argv, preprocessor, model_api, config):
        super(Tester, self).__init__(argv, preprocessor, model_api, config)

    def _setup_word(self):
        self.vocab_word = self.load_data(self.argv.load_word)

    def _setup_label(self):
        self.vocab_label = self.load_data(self.argv.load_label)

    def _setup_samples(self):
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]

    def _setup_model_api(self):
        self.model_api.compile(vocab_word=self.vocab_word, vocab_label=self.vocab_label, init_emb=None)
        self.model_api.load_params(self.argv.load_param)
        self.model_api.set_predict_f()

    def predict(self):
        model_api = self.model_api

        if self.dev_samples:
            print '\n  DEV\n\t',
            dev_results = model_api.predict_one_epoch(self.dev_samples)
            dev_f1 = model_api.eval_one_epoch(dev_results.decoder_outputs, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if self.test_samples:
            print '\n  TEST\n\t',
            test_results = model_api.predict_one_epoch(self.test_samples)
            test_f1 = model_api.eval_one_epoch(test_results.decoder_outputs, self.test_samples)
            say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))
            if self.argv.save:
                model_api.save_pas_results(results=test_results.decoder_outputs, samples=self.test_samples)


class JackKnifeTester(Tester):

    def __init__(self, argv, preprocessor, model_api, config):
        super(JackKnifeTester, self).__init__(argv, preprocessor, model_api, config)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        train_set = self.load_data(self.argv.train_data)
        train_corpus, test_corpus = self.separate_train_part(train_set, self.argv.sec)
        dev_corpus = self.preprocessor.corpus_loader.load_corpus(self.argv.dev_data)
        return train_corpus, dev_corpus, test_corpus

    @staticmethod
    def separate_train_part(train_set, sec):
        return None, train_set[sec]

    def predict(self):
        argv = self.argv
        model_api = self.model_api

        if self.dev_samples:
            print '\n  DEV\n\t',
            dev_results, dev_results_prob = model_api.predict_one_epoch(self.dev_samples)
            dev_f1 = model_api.eval_one_epoch(dev_results, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if self.test_samples:
            print '\n  TEST\n\t',
            test_results, test_results_prob = model_api.predict_one_epoch(self.test_samples)
            test_f1 = model_api.eval_one_epoch(test_results, self.test_samples)
            say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))
            test_n_best_lists = model_api.create_n_best_lists(self.test_samples, test_results_prob)
            model_api.eval_n_best_lists(self.test_samples, test_n_best_lists)

            sec = 'all' if argv.sec is None else str(argv.sec)
            output_fn = 'sec-%s' % sec
            output_dir = 'data/rerank/list/layers%d/best%d/' % (model_api.argv.layers, model_api.argv.n_best)
            if argv.output_dir is not None:
                output_dir = argv.output_dir
            model_api.save_n_best_lists(output_fn, output_dir, test_n_best_lists)


class NBestTester(JackKnifeTester):

    def __init__(self, argv, preprocessor, model_api, config):
        super(NBestTester, self).__init__(argv, preprocessor, model_api, config)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        dev_corpus = self.preprocessor.corpus_loader.load_corpus(self.argv.dev_data)
        test_corpus = self.preprocessor.corpus_loader.load_corpus(self.argv.test_data)
        return None, dev_corpus, test_corpus
