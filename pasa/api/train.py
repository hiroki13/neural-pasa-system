import os
import shutil

import numpy as np
import theano

from abc import ABCMeta
from ..utils.io_utils import say, dump_data, load_data, move_data
from ..utils.preprocessor import Preprocessor, RankingPreprocessor, RerankingPreprocessor
from ..ling.vocab import Vocab, PAD, UNK
from ..model.model_api import ModelAPI, RankingModelAPI, NBestModelAPI, RerankingModelAPI


class Trainer(object):
    __metaclass__ = ABCMeta

    def __init__(self, argv, preprocessor):
        self.argv = argv
        self.preprocessor = preprocessor
        self.model_api = None

        self.vocab_word = None
        self.vocab_label = None
        self.trainable_emb = None
        self.untrainable_emb = None

        self.corpus_set = None
        self.train_samples = None
        self.dev_samples = None
        self.test_samples = None
        self.train_indices = None

    def setup_training(self):
        say('\n\nSETTING UP AN INTRA-SENTENTIAL PASA TRAINING SETTING\n')
        self._setup_corpus()
        self._setup_vocab_word()
        self._setup_label()
        if self.argv.save:
            self._save()
        self._setup_samples()

    def _setup_corpus(self):
        self.corpus_set = self._load_corpus_set()
        self._show_corpus_stats(self.corpus_set)

    def _load_corpus_set(self):
        self.preprocessor.set_corpus_loader()
        return self.preprocessor.load_corpus_set()

    def _show_corpus_stats(self, corpus_set):
        self.preprocessor.show_corpus_stats(corpus_set)

    def _setup_vocab_word(self):
        vocab_word, emb = self.preprocessor.load_init_emb()

        if emb is None:
            vocab_word = self.preprocessor.create_vocab_word(self.corpus_set[0])
        elif emb is not None and self.argv.fix == 0:
            vocab_word, emb, untrainable_emb = self._divide_emb(self.corpus_set[0], vocab_word, emb)
            self.untrainable_emb = untrainable_emb

        self.vocab_word = vocab_word
        self.trainable_emb = emb

    def _divide_emb(self, corpus, vocab_emb_word, emb):
        say('\nDivide the embeddings into the trainable/untrainable embeddings\n')

        vocab_trainable_word = self._get_common_vocab(corpus, vocab_emb_word)
        vocab_trainable_word.add_word(UNK)
        vocab_untrainable_word = Vocab()
        trainable_emb = [[] for i in xrange(vocab_trainable_word.size())]
        untrainable_emb = []

        for w, w_id in vocab_emb_word.w2i.items():
            # TODO: change 'PAD' to '<PAD>'
            if w == PAD:
                continue
            if vocab_trainable_word.has_key(w):
                trainable_emb[vocab_trainable_word.get_id(w)] = emb[w_id]
            else:
                untrainable_emb.append(emb[w_id])
                vocab_untrainable_word.add_word(w)

        vocab_word = self._unite_vocab_word(vocab_trainable_word, vocab_untrainable_word)
        trainable_emb = np.asarray(trainable_emb, theano.config.floatX)
        untrainable_emb = np.asarray(untrainable_emb, theano.config.floatX)

        say('\tTrainable emb: %d  Untrainable emb: %d\n' % (trainable_emb.shape[0], untrainable_emb.shape[0]))
        say('Vocab size: %d  Trainable: %d  Untrainable: %d' % (vocab_word.size(),
                                                                vocab_trainable_word.size(),
                                                                vocab_untrainable_word.size()))
        assert vocab_word.size() == (trainable_emb.shape[0] + untrainable_emb.shape[0] + 1)

        return vocab_word, trainable_emb, untrainable_emb

    @staticmethod
    def _get_common_vocab(corpus, vocab_emb_word):
        vocab_word = Vocab()
        for doc in corpus:
            for sent in doc:
                for w in sent:
                    if vocab_emb_word.has_key(w.form):
                        vocab_word.add_word(w.form)
        return vocab_word

    @staticmethod
    def _unite_vocab_word(vocab_word_1, vocab_word_2):
        vocab_word = Vocab()
        vocab_word.set_init_word()
        for w in vocab_word_1.i2w:
            vocab_word.add_word(w)
        for w in vocab_word_2.i2w:
            vocab_word.add_word(w)
        return vocab_word

    def _setup_label(self):
        self.vocab_label = self.preprocessor.create_vocab_label()

    def _setup_samples(self):
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        train_sample_shared, train_batch_index = self.preprocessor.create_shared_samples(sample_set[0])
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.train_samples = train_sample_shared
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]
        self.train_indices = train_batch_index

    def _save(self):
        output_fn = 'vocab_word.cut-%d' % self.argv.vocab_cut_off
        dump_data(self.vocab_word, output_fn)
        move_data(output_fn + '.pkl.gz', 'data/word')
        dump_data(self.vocab_label, 'vocab_label')
        move_data('vocab_label.pkl.gz', 'data/label')

    def train_model(self):
        say('\n\nTRAINING A MODEL\n')
        model_api = self.model_api = ModelAPI(argv=self.argv,
                                              emb=self.trainable_emb,
                                              vocab_word=self.vocab_word,
                                              vocab_label=self.vocab_label)

        model_api.compile(train_sample_shared=self.train_samples)

        model_api.train_all(argv=self.argv,
                            train_batch_index=self.train_indices,
                            dev_samples=self.dev_samples,
                            test_samples=self.test_samples,
                            untrainable_emb=self.untrainable_emb)


class RankingTrainer(Trainer):
    def __init__(self, argv, preprocessor):
        super(RankingTrainer, self).__init__(argv, preprocessor)

    def _setup_samples(self):
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)
        train_sample_shared = self.preprocessor.create_shared_samples(sample_set[0])
        self.preprocessor.show_sample_stats(sample_set, self.vocab_label)

        self.train_samples = train_sample_shared
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]

    def train_model(self):
        say('\n\nTRAINING A MODEL\n')
        model_api = RankingModelAPI(argv=self.argv,
                                    emb=self.trainable_emb,
                                    vocab_word=self.vocab_word,
                                    vocab_label=self.vocab_label)

        model_api.compile()

        model_api.train_all(argv=self.argv,
                            train_samples=self.train_samples,
                            dev_samples=self.dev_samples,
                            test_samples=self.test_samples,
                            untrainable_emb=self.untrainable_emb)


class TrainCorpusSeparator(Trainer):
    def __init__(self, argv, preprocessor):
        super(TrainCorpusSeparator, self).__init__(argv, preprocessor)

    def setup_training(self):
        say('\n\nSeparating the training set\n')
        self._setup_corpus()
        train_set = self.separate_train_data(10)
        self.save_train_samples(train_set)

    def train_model(self):
        pass

    def separate_train_data(self, n_seps):
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

    @staticmethod
    def save_train_samples(train_set):
        if not os.path.exists('data/rerank'):
            os.mkdir('data/rerank')
        if not os.path.exists('data/rerank/train'):
            os.mkdir('data/rerank/train')
        output_fn = 'train.%d.pkl.gz' % len(train_set)
        output_dir = 'data/rerank/train/'
        dump_data(train_set, output_fn)
        move_data(output_fn, output_dir)


class NBestTrainer(Trainer):
    def __init__(self, argv, preprocessor):
        super(NBestTrainer, self).__init__(argv, preprocessor)

    def train_model(self):
        say('\n\nTRAINING AN N-BEST MODEL\n')
        model_api = self.model_api = NBestModelAPI(argv=self.argv,
                                                   emb=self.trainable_emb,
                                                   vocab_word=self.vocab_word,
                                                   vocab_label=self.vocab_label)

        model_api.compile(train_sample_shared=self.train_samples)

        model_api.train_all(argv=self.argv,
                            train_batch_index=self.train_indices,
                            dev_samples=self.dev_samples,
                            test_samples=self.test_samples,
                            untrainable_emb=self.untrainable_emb)


class JackKnifeTrainer(NBestTrainer):
    def __init__(self, argv, preprocessor):
        super(JackKnifeTrainer, self).__init__(argv, preprocessor)

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

    def _save(self):
        output_fn = 'vocab_word.cut-%d.n_best-%d' % (self.argv.vocab_cut_off, self.argv.n_best)
        if self.argv.sec is None:
            output_fn += '.all'
        else:
            output_fn += '.sec-%d' % self.argv.sec
        dump_data(self.vocab_word, output_fn)
        move_data(output_fn + '.pkl.gz', 'data/rerank/word')

        dump_data(self.vocab_label, 'vocab_label')
        move_data('vocab_label.pkl.gz', 'data/rerank/label')


class RerankingTrainer(Trainer):
    def __init__(self, argv, preprocessor):
        super(RerankingTrainer, self).__init__(argv, preprocessor)

    def setup_training(self):
        say('\n\nSETTING UP AN INTRA-SENTENTIAL PASA TRAINING SETTING\n')
        self._setup_corpus()
        self._setup_vocab_word()
        self._setup_label()
        self._setup_samples()

    def _load_corpus_set(self):
        train_corpus = load_data(self.argv.train_data)
        dev_corpus = load_data(self.argv.dev_data)
        test_corpus = load_data(self.argv.test_data)
        return train_corpus, dev_corpus, test_corpus

    def _show_corpus_stats(self, corpus_set):
        pass

    def _setup_samples(self):
        self.preprocessor.set_sample_factory(self.vocab_word, self.vocab_label)
        sample_set = self.preprocessor.create_sample_set(self.corpus_set)

        self.train_samples = self.preprocessor.create_shared_samples(sample_set[0])
        self.dev_samples = sample_set[1]
        self.test_samples = sample_set[2]

    def train_model(self):
        say('\n\nTRAINING AN RERANKING MODEL\n')
        model_api = self.model_api = RerankingModelAPI(argv=self.argv,
                                                       emb=self.trainable_emb,
                                                       vocab_word=self.vocab_word,
                                                       vocab_label=self.vocab_label)

        model_api.compile()
        model_api.train_all(argv=self.argv,
                            train_samples=self.train_samples,
                            dev_samples=self.dev_samples,
                            test_samples=self.test_samples,
                            untrainable_emb=self.untrainable_emb)


def select_trainer(argv):
    if argv.model == 'base':
        return Trainer(argv, Preprocessor(argv))
    elif argv.model == 'rank':
        return RankingTrainer(argv, RankingPreprocessor(argv))
    elif argv.model == 'jack':
        return JackKnifeTrainer(argv, Preprocessor(argv))
    elif argv.model == 'sep':
        return TrainCorpusSeparator(argv, Preprocessor(argv))
    elif argv.model == 'rerank':
        return RerankingTrainer(argv, RerankingPreprocessor(argv))
    return NBestTrainer(argv, Preprocessor(argv))


def main(argv):
    trainer = select_trainer(argv)
    trainer.setup_training()
    trainer.train_model()
