from experimenter import Experimenter

from ..utils.io_utils import say, dump_data, move_data


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

        self.train_samples = self.preprocessor.create_batch(sample_set[0])
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]
        say('\nMini-Batches: %d\n\n' % (self.train_samples.size()))

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
        train_set = self.load_data(self.argv.train_data)
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

    def _setup_samples(self):
        say('\n\nSetting up samples...\n')
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set, self.vocab_word = self.preprocessor.create_sample_set(self.corpus_set)
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.train_samples = self.preprocessor.create_batch(sample_set[0], 5)
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]
        say('\nMini-Batches: %d\n\n' % (len(self.train_samples)))


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


