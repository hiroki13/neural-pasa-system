import os
from abc import ABCMeta, abstractmethod

from ..utils.io_utils import say, dump_data, move_data, load_data


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
        self.preprocessor.show_corpus_stats(self.corpus_set)

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

