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
        dev_corpus, word_freqs = io_utils.load_ntc(path=argv.dev_data, data_size=data_size, model='word',
                                                   word_freqs=word_freqs)
        print '\nDEV',
        corpus_statistics(dev_corpus)
    else:
        dev_corpus = None

    if argv.test_data:
        test_corpus, word_freqs = io_utils.load_ntc(path=argv.test_data, data_size=data_size, model='word',
                                                    word_freqs=word_freqs)
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
    #############
    # Set vocab #
    #############
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


def train(argv, model_api, train_batch_index, dev_samples, test_samples):
    print '\nTRAINING START\n'
    n_train_batches = len(train_batch_index)
    tr_indices = range(n_train_batches)

    f1_history = {}
    best_dev_f1 = -1.

    for epoch in xrange(argv.epoch):
        train_eval = Eval()
        dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
        model_api.model.dropout.set_value(dropout_p)

        print '\nEpoch: %d' % (epoch + 1)
        print '  TRAIN\n\t',

        np.random.shuffle(tr_indices)
        start = time.time()

        ############
        # Training #
        ############
        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            batch_range = train_batch_index[b_index]
            result_sys, result_gold, nll = model_api.train(index=b_index, bos=batch_range[0], eos=batch_range[1])

            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_results()

        ###############
        # Development #
        ###############
        update = False
        if argv.dev_data:
            print '\n  DEV\n\t',
            dev_f1 = model_api.predict_all(dev_samples)
            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                f1_history[epoch+1] = [best_dev_f1]
                update = True

                if argv.save:
                    model_api.save_model('model.intra.layers-%d.window-%d.reg-%f' % (argv.layers, argv.window, argv.reg))
                    model_api.output_results('result.dev.intra.layers-%d.window-%d.reg-%f.txt' %
                                           (argv.layers, argv.window, argv.reg),
                                           dev_samples)

        ########
        # Test #
        ########
        if argv.test_data:
            print '\n  TEST\n\t',
            test_f1 = model_api.predict_all(test_samples)
            if update:
                model_api.output_results('result.test.intra.layer-%d.window-%d.reg-%f.txt' %
                                       (argv.layer, argv.window, argv.reg),
                                       test_samples)
                if epoch+1 in f1_history:
                    f1_history[epoch+1].append(test_f1)
                else:
                    f1_history[epoch+1] = [test_f1]

        ###########
        # Results #
        ###########
        say('\n\n\tF1 HISTORY')
        for k, v in sorted(f1_history.items()):
            if len(v) == 2:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
            else:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
        say('\n\n')


def main(argv):
    say('\n\nSETTING UP AN INTRA-SENTENTIAL TRAINING SETTING\n')

    train_corpus, dev_corpus, test_corpus, word_freqs = get_corpus(argv)
    vocab_label = set_labels(argv)
    vocab_word = set_vocab(argv, word_freqs)
    train_samples, dev_samples, test_samples = create_samples(argv, train_corpus, dev_corpus, test_corpus,
                                                              vocab_word, vocab_label)
    train_sample_shared, train_batch_index = create_shared_samples(argv, train_samples)
    model_api = set_model(argv, train_sample_shared, vocab_word, vocab_label)

    train(argv, model_api, train_batch_index, dev_samples, test_samples)
