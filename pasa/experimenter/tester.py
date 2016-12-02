from experimenter import Experimenter

from ..utils.io_utils import say


class Tester(Experimenter):

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

