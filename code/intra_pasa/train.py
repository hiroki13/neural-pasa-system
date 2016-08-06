import sys
import time
import math

import theano
import numpy as np

from utils import io_utils
from utils.io_utils import say
from ling.vocab import Vocab
from preprocessor import get_samples, theano_format, theano_format_online
from stats.stats import corpus_statistics, sample_statistics, check_samples
from model import Model
from eval import Eval


def train(argv):
    say('\n\nSETTING UP AN INTRA-SENTENTIAL TRAINING SETTING\n')

    emb = None

    ##############
    # Set labels #
    ##############
    vocab_label = Vocab()
    vocab_label.set_pas_labels()
    print '\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i))

    ##############
    # Load files #
    ##############
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    train_corpus, word_freqs = io_utils.load_ntc(path=argv.train_data, data_size=argv.data_size, model='word')
    print '\nTRAIN',
    corpus_statistics(train_corpus)

    if argv.dev_data:
        dev_corpus, word_freqs = io_utils.load_ntc(path=argv.dev_data, data_size=argv.data_size, model='word',
                                                   word_freqs=word_freqs)
        print '\nDEV',
        corpus_statistics(dev_corpus)

    if argv.test_data:
        test_corpus, word_freqs = io_utils.load_ntc(path=argv.test_data, data_size=argv.data_size, model='word',
                                                    word_freqs=word_freqs)
        print '\nTEST',
        corpus_statistics(test_corpus)

    #############
    # Set vocab #
    #############
    vocab_word = Vocab()
    vocab_word.set_init_word()
    vocab_word.add_vocab(word_freqs=word_freqs, vocab_cut_off=argv.vocab_cut_off)
    print '\nVocab: %d\tType: word\n' % vocab_word.size()

    #################
    # Preprocessing #
    #################
    # pre_samples: (x, y)
    # x: 1D: n_sents, 2D: n_prds, 3D: n_words, 4D: window + 2
    # y: 1D: n_sents, 2D: n_prds, 3D: n_words
    # word_ids: 1D: n_sents, 2D: n_words

    tr_pre_samples, tr_word_ids = get_samples(train_corpus, vocab_word, vocab_label, argv.window)
    tr_samples, train_batch_index = theano_format(tr_pre_samples)
    print '\nTRAIN',
    sample_statistics(tr_pre_samples[1], vocab_label)
    n_train_batches = len(train_batch_index)
    print '\tTrain Mini-Batches: %d\n' % n_train_batches

    if argv.dev_data:
        dev_pre_samples, dev_word_ids = get_samples(dev_corpus, vocab_word, vocab_label, argv.window)
#        dev_samples, dev_batch_index = theano_format(dev_pre_samples)
        dev_samples = theano_format_online(dev_pre_samples)
#        n_dev_batches = len(dev_batch_index)
        n_dev_batches = len(dev_samples[-1])
        print '\nDEV',
        sample_statistics(dev_pre_samples[1], vocab_label)
        print '\tDev Mini-Batches: %d\n' % n_dev_batches

    if argv.test_data:
        test_pre_samples, test_word_ids = get_samples(test_corpus, vocab_word, vocab_label, argv.window)
#        test_samples, test_batch_index = theano_format(test_pre_samples)
        test_samples = theano_format_online(test_pre_samples)
#        n_test_batches = len(test_batch_index)
        n_test_batches = len(test_samples[-1])
        print '\nTEST',
        sample_statistics(test_pre_samples[1], vocab_label)
        print '\tTest Mini-Batches: %d\n' % n_test_batches

    if argv.check:
        check_samples(tr_pre_samples, vocab_word, vocab_label)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '\n\nBuilding a model...'
    model = Model(argv=argv, emb=emb, vocab_word=vocab_word, vocab_label=vocab_label)
    model.compile()

    model.set_train_f(tr_samples)
    if argv.dev_data or argv.test_data:
        model.set_predict_f()

    ###############
    # TRAIN MODEL #
    ###############
    print '\nTRAINING START\n'

    tr_indices = range(n_train_batches)

    best_dev_f1 = -1.
    best_test_f1 = -1.

    for epoch in xrange(argv.epoch):
        train_eval = Eval()
        dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
        model.dropout.set_value(dropout_p)

        print '\nEpoch: %d' % (epoch + 1)
        print '  TRAIN\n\t',

        np.random.shuffle(tr_indices)
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            batch_range = train_batch_index[b_index]
            result_sys, result_gold, nll = model.train(index=b_index, bos=batch_range[0], eos=batch_range[1])

            assert not math.isnan(nll), 'NLL is NAN: Index: %d' % index

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\tTime: %f' % (time.time() - start)
        train_eval.show_results()

        ##############
        # Validating #
        ##############
        update = False
        if argv.dev_data:
            print '\n  DEV\n\t',
            dev_f1 = predict(model, dev_samples)
            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                update = True

        if argv.test_data:
            print '\n  TEST\n\t',
            test_f1 = predict(model, test_samples)
            if update:
                best_test_f1 = test_f1

        say('\n\n\tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}\n'.format(best_dev_f1, best_test_f1))


def predict(model, samples):
    pred_eval = Eval()
    start = time.time()
    model.dropout.set_value(0.0)

    for index in xrange(len(samples[0])):
        if index != 0 and index % 1000 == 0:
            print index,
            sys.stdout.flush()

        x = samples[0][index]
        y = samples[1][index]
        sent_len = samples[2][index]

        results_sys, results_gold = model.predict(x, y, sent_len)
        pred_eval.update_results(results_sys, results_gold)

    print '\tTime: %f' % (time.time() - start)
    pred_eval.show_results()

    return pred_eval.all_f1


def main(argv):
    train(argv)
