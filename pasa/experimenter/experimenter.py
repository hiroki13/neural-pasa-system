import os
from abc import ABCMeta, abstractmethod

from ..utils.io_utils import say, dump_data, move_data, load_data
from ..utils.stats import corpus_statistics, sample_statistics, show_case_dist


class Experimenter(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv, preprocessor, model_api, epoch_manager=None, config=None):
        self.argv = argv
        self.config = config
        self.preprocessor = preprocessor
        self.model_api = model_api
        self.epoch_manager = epoch_manager

        self.vocab_word = None
        self.vocab_label = None
        self.trainable_emb = None
        self.untrainable_emb = None

        self.corpus_set = None
        self.train_samples = None
        self.dev_samples = None
        self.test_samples = None

        self.output_path = 'data/%s' % self.argv.model

    def setup_experiment(self):
        say('\n\nSETTING UP A PASA EXPERIMENT\n')
        self._setup_corpus()
        self._setup_word()
        self._setup_label()
        self._setup_samples()
        self._setup_model_api()

    def _setup_corpus(self):
        self.corpus_set = self.preprocessor.load_corpus_set()
        self._show_corpus_stats(self.corpus_set)

    @abstractmethod
    def _setup_word(self):
        raise NotImplementedError()

    @abstractmethod
    def _setup_label(self):
        raise NotImplementedError()

    @abstractmethod
    def _setup_samples(self):
        raise NotImplementedError()

    @abstractmethod
    def _setup_model_api(self):
        raise NotImplementedError()

    def save_word(self):
        output_fn = 'vocab_word.model-%s.cut-%d' % (self.argv.model, self.argv.vocab_cut_off)
        output_path = self.output_path + '/word'
        self._create_path(output_path)
        dump_data(self.vocab_word, output_fn)
        move_data(output_fn + '.pkl.gz', output_path)

    def save_label(self):
        output_fn = 'vocab_label.model-%s' % self.argv.model
        output_path = self.output_path + '/label'
        self._create_path(output_path)
        dump_data(self.vocab_label, output_fn)
        move_data(output_fn + '.pkl.gz', output_path)

    @staticmethod
    def load_data(fn):
        return load_data(fn)

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
    def _show_corpus_stats(corpora):
        for corpus in corpora:
            corpus_statistics(corpus)
            show_case_dist(corpus)

    @staticmethod
    def _show_sample_stats(sample_set, vocab_label):
        for samples in sample_set:
            sample_statistics(samples, vocab_label)


class Trainer(Experimenter):

    def _setup_word(self):
        say('\n\nSetting up vocabularies...\n')
        pp = self.preprocessor
        vocab_word, emb = pp.load_init_emb()
        vocab_word, trainable_emb, untrainable_emb = pp.create_trainable_emb(train_corpus=self.corpus_set[0],
                                                                             vocab_word=vocab_word,
                                                                             emb=emb)
        self.vocab_word = vocab_word
        self.trainable_emb = trainable_emb
        self.untrainable_emb = untrainable_emb

        if self.argv.save:
            self.save_word()

    def _setup_label(self):
        say('\n\nSetting up target labels...\n')
        self.vocab_label = self.preprocessor.create_vocab_label()
        if self.argv.save:
            self.save_label()

    def _setup_samples(self):
        say('\n\nSetting up samples...\n')
        pp = self.preprocessor
        pp.set_sample_factory(self.vocab_word, self.vocab_label)

        sample_set = pp.create_sample_set(self.corpus_set)
        self.train_samples = pp.create_batches(sample_set[0])
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]

        self._show_sample_stats(sample_set, self.vocab_label)
        say('\nMini-Batches: %d\n\n' % (self.train_samples.size()))

    def _setup_model_api(self):
        say('\n\nSetting up a model API...\n')
        self.model_api.compile(vocab_word=self.vocab_word,
                               vocab_label=self.vocab_label,
                               init_emb=self.trainable_emb)
        self.model_api.set_train_f()
        self.model_api.set_predict_f()

    def train(self):
        say('\n\nTRAINING START\n\n')
        self.epoch_manager.train(model_api=self.model_api,
                                 train_samples=self.train_samples,
                                 dev_samples=self.dev_samples,
                                 test_samples=self.test_samples,
                                 untrainable_emb=self.untrainable_emb)


class Tester(Experimenter):

    def _setup_word(self):
        self.vocab_word = self.load_data(self.argv.load_word)

    def _setup_label(self):
        self.vocab_label = self.load_data(self.argv.load_label)

    def _setup_samples(self):
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)

        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]

        self._show_sample_stats(sample_set, self.vocab_label)

    def _setup_model_api(self):
        self.model_api.compile(vocab_word=self.vocab_word, vocab_label=self.vocab_label, init_emb=None)
        self.model_api.load_params(self.argv.load_param)
        self.model_api.set_predict_f()

    def predict(self):
        model_api = self.model_api

        if self.dev_samples:
            print '\n  DEV\n\t',
            dev_results = model_api.predict_one_epoch(self.dev_samples)
            dev_f1 = model_api.eval_one_epoch(dev_results, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if self.test_samples:
            print '\n  TEST\n\t',
            test_results = model_api.predict_one_epoch(self.test_samples)
            test_f1 = model_api.eval_one_epoch(test_results, self.test_samples)
            say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))
#            if self.argv.save:
#                model_api.save_pas_results(results=test_results.decoder_outputs, samples=self.test_samples)


