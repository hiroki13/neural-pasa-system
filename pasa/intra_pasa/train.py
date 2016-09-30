import sys
import time
import math

import theano
import numpy as np

from ..utils import io_utils
from ..utils.io_utils import say, dump_data
from ..ling.vocab import Vocab
from ..stats.stats import corpus_statistics, sample_statistics, check_samples
from preprocessor import get_samples, get_shared_samples
from model_api import ModelAPI
from eval import Eval


def get_corpus(argv):
    data_size = argv.data_size

    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    train_corpus, word_freqs = io_utils.load_ntc(path=argv.train_data, data_size=data_size, model='word')
    print '\nTRAIN',
    corpus_statistics(train_corpus)

    if argv.dev_data:
        dev_corpus, _ = io_utils.load_ntc(path=argv.dev_data, data_size=data_size, model='word')
        print '\nDEV',
        corpus_statistics(dev_corpus)
    else:
        dev_corpus = None

    if argv.test_data:
        test_corpus, _ = io_utils.load_ntc(path=argv.test_data, data_size=data_size, model='word')
        print '\nTEST',
        corpus_statistics(test_corpus)
    else:
        test_corpus = None

    return train_corpus, dev_corpus, test_corpus, word_freqs


def set_labels(argv):
    vocab_label = Vocab()
    vocab_label.set_pas_labels()
    if argv.save:
        dump_data(vocab_label, 'vocab_label')
    print '\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i))
    return vocab_label


def set_vocab(argv, word_freqs):
    vocab_word = Vocab()
    vocab_word.set_init_word()
    vocab_word.add_vocab(word_freqs=word_freqs, vocab_cut_off=argv.vocab_cut_off)
    if argv.save:
        dump_data(vocab_word, 'vocab_word.cut-%d' % argv.vocab_cut_off)
    print '\nVocab: %d\tType: word\n' % vocab_word.size()
    return vocab_word


def create_samples(argv, train_corpus, dev_corpus, test_corpus, vocab_word, vocab_label):
    window = argv.window

    # samples: 1D: n_sents; Sample
    train_samples = get_samples(train_corpus, vocab_word, vocab_label, window)
    print '\nTRAIN',
    sample_statistics(train_samples, vocab_label)

    if dev_corpus:
        dev_samples = get_samples(dev_corpus, vocab_word, vocab_label, window, test=True)
        print '\nDEV',
        sample_statistics(dev_samples, vocab_label)
    else:
        dev_samples = None

    if test_corpus:
        test_samples = get_samples(test_corpus, vocab_word, vocab_label, window, test=True)
        print '\nTEST',
        sample_statistics(test_samples, vocab_label)
    else:
        test_samples = None

    if argv.check:
        check_samples(train_samples, vocab_word, vocab_label)

    return train_samples, dev_samples, test_samples


def create_shared_samples(argv, train_samples):
    mp = True if argv.attention else False

    # samples: 1D: n_sents; Sample
    train_sample_shared, train_batch_index = get_shared_samples(train_samples, batch_size=argv.batch_size, mp=mp)
    return train_sample_shared, train_batch_index


def set_model(argv, train_sample_shared, vocab_word, vocab_label):
    emb = None
    print '\n\nBuilding a model...'
    model_api = ModelAPI(argv=argv, emb=emb, vocab_word=vocab_word, vocab_label=vocab_label)
    model_api.set_model()

    model_api.set_train_f(train_sample_shared)
    if argv.dev_data or argv.test_data:
        model_api.set_predict_f()

    return model_api


def main(argv):
    say('\n\nSETTING UP AN INTRA-SENTENTIAL PASA TRAINING SETTING\n')

    train_corpus, dev_corpus, test_corpus, word_freqs = get_corpus(argv)
    vocab_label = set_labels(argv)
    vocab_word = set_vocab(argv, word_freqs)
    train_samples, dev_samples, test_samples = create_samples(argv, train_corpus, dev_corpus, test_corpus,
                                                              vocab_word, vocab_label)
    train_sample_shared, train_batch_index = create_shared_samples(argv, train_samples)
    model_api = set_model(argv, train_sample_shared, vocab_word, vocab_label)

    model_api.train_all(argv, train_batch_index, dev_samples, test_samples)
