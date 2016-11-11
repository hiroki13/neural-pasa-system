import os
from abc import ABCMeta, abstractmethod

from ..utils.io_utils import say, dump_data, load_data, move_data, load_dir


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
        self.corpus_set = self._load_corpus_set()
        self._show_corpus_stats(self.corpus_set)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        return self.preprocessor.load_corpus_set()

    def _show_corpus_stats(self, corpus_set):
        self.preprocessor.show_corpus_stats(corpus_set)

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
    def _create_path(output_path):
        path = ''
        dir_names = output_path.split('/')
        for dir_name in dir_names:
            path += dir_name
            if not os.path.exists(path):
                os.mkdir(path)
            path += '/'


class Trainer(Experimenter):

    def __init__(self, argv, preprocessor, model_api, epoch_manager):
        super(Trainer, self).__init__(argv, preprocessor, model_api, epoch_manager, None)

    def _setup_word(self):
        say('\n\nSetting up vocabularies...\n')
        vocab_word, emb = self.preprocessor.load_init_emb()
        vocab_word, emb, untrainable_emb = self.preprocessor.create_trainable_emb(train_corpus=self.corpus_set[0],
                                                                                  vocab_word=vocab_word,
                                                                                  emb=emb)
        self.vocab_word = vocab_word
        self.trainable_emb = emb
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
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.train_samples = self.preprocessor.create_batched_samples(sample_set[0], 3)
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]
        say('\nMini-Batches: %d\n\n' % (len(self.train_samples)))

    def _setup_model_api(self):
        say('\n\nSetting up a model API...\n')
        self.model_api.compile(vocab_word=self.vocab_word, vocab_label=self.vocab_label, init_emb=self.trainable_emb)
        self.model_api.set_train_f()
        self.model_api.set_predict_f()

    def train(self):
        say('\n\nTRAINING START\n\n')
        self.epoch_manager.train(model_api=self.model_api,
                                 train_samples=self.train_samples,
                                 dev_samples=self.dev_samples,
                                 test_samples=self.test_samples,
                                 untrainable_emb=self.untrainable_emb)


class JackKnifeTrainer(Trainer):

    def __init__(self, argv, preprocessor, model_api, epoch_manager):
        super(JackKnifeTrainer, self).__init__(argv, preprocessor, model_api, epoch_manager)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        train_set = load_data(self.argv.train_data)
        train_corpus, test_corpus = self.separate_train_part(train_set, self.argv.sec)
        dev_corpus = self.preprocessor.corpus_loader.load_corpus(self.argv.dev_data)
        return train_corpus, dev_corpus, test_corpus

    @staticmethod
    def separate_train_part(train_set, sec):
        train_part = []
        for i, one_train in enumerate(train_set):
            if i == sec:
                continue
            train_part.extend(one_train)
        return train_part, train_set[sec]

    def save_word(self):
        output_fn = 'vocab_word.model-%s.cut-%d' % (self.argv.model, self.argv.vocab_cut_off)
        if self.argv.sec is None:
            output_fn += '.all'
        else:
            output_fn += '.sec-%d' % self.argv.sec
        output_path = self.output_path + '/word'
        self._create_path(output_path)
        dump_data(self.vocab_word, output_fn)
        move_data(output_fn + '.pkl.gz', output_path)


class StackingTrainer(Trainer):

    def __init__(self, argv, preprocessor, model_api, epoch_manager):
        super(StackingTrainer, self).__init__(argv, preprocessor, model_api, epoch_manager)

    def _setup_word(self):
        pass


class RerankingTrainer(Trainer):

    def __init__(self, argv, preprocessor, model_api, config):
        super(RerankingTrainer, self).__init__(argv, preprocessor, model_api, config)

    def _load_corpus_set(self):
        train_corpus = load_dir(self.argv.train_data)
        dev_corpus = load_data(self.argv.dev_data)
        test_corpus = load_data(self.argv.test_data)
        return train_corpus, dev_corpus, test_corpus

    def _show_corpus_stats(self, corpus_set):
        pass


class TrainCorpusSeparator(Trainer):

    def __init__(self, argv, preprocessor, model_api, epoch_manager):
        super(TrainCorpusSeparator, self).__init__(argv, preprocessor, model_api, epoch_manager)

    def setup_experiment(self):
        self._setup_corpus()
        train_set = self._separate_train_data(10)
        self._save_train_samples(train_set)

    def train(self):
        pass

    def _separate_train_data(self, n_seps):
        train_corpus = self.corpus_set[0]
        n_samples = len(train_corpus)
        slide = n_samples / n_seps
        separated_train_set = []
        for i in xrange(n_seps - 1):
            one_train = train_corpus[i * slide: (i + 1) * slide]
            separated_train_set.append(one_train)
        one_train = train_corpus[(n_seps - 1) * slide:]
        separated_train_set.append(one_train)
        return separated_train_set

    def _save_train_samples(self, train_set):
        output_fn = 'train.%d.pkl.gz' % len(train_set)
        output_path = self.output_path + '/train'
        self._create_path(output_path)
        dump_data(train_set, output_fn)
        move_data(output_fn, output_path)


class Tester(Experimenter):

    def __init__(self, argv, preprocessor, model_api, config):
        super(Tester, self).__init__(argv, preprocessor, model_api, config)

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
        self.model_api.compile(vocab_word=self.vocab_word, vocab_label=self.vocab_label, init_emb=None)
        self.model_api.load_params(self.argv.load_params)
        self.model_api.set_predict_f()

    def predict(self):
        model_api = self.model_api

        if self.argv.dev_data:
            print '\n  DEV\n\t',
            dev_results, dev_results_prob = model_api.predict_all(self.dev_samples)
            dev_f1 = model_api.eval_all(dev_results, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if self.argv.test_data:
            print '\n  TEST\n\t',
            test_results, test_results_prob = model_api.predict_all(self.test_samples)
            test_f1 = model_api.eval_all(test_results, self.test_samples)
            say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))


class JackKnifeTester(Tester):

    def __init__(self, argv, preprocessor, model_api, config):
        super(JackKnifeTester, self).__init__(argv, preprocessor, model_api, config)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        train_set = load_data(self.argv.train_data)
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
            dev_results, dev_results_prob = model_api.predict_all(self.dev_samples)
            dev_f1 = model_api.eval_all(dev_results, self.dev_samples)
            say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

        if self.test_samples:
            print '\n  TEST\n\t',
            test_results, test_results_prob = model_api.predict_all(self.test_samples)
            test_f1 = model_api.eval_all(test_results, self.test_samples)
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
