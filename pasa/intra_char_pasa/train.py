import sys
import time
import numpy as np

from utils import io_utils
from ling.vocab import Vocab
from preprocessor import get_samples, theano_format
from stats.stats import corpus_statistics, sample_statistics, check_samples
from model_builder import set_model, set_train_f, set_pred_f
from eval import Eval


def train(argv):
    print '\nSETTING UP AN INTRA-SENTENTIAL TRAINING SETTING\n'

    emb = None

    """ Set labels """
    vocab_label = Vocab()
    vocab_label.set_pas_labels_char()
    print '\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i))

    ##############
    # Load files #
    ##############
    # corpus: 1D: n_sents, 2D: n_chars, 3D: (char, pas_info, pas_id)
    train_corpus, word_freqs = io_utils.load_ntc(path=argv.train_data, data_size=argv.data_size, model='char')
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
    print '\nVocab: %d\tType: char\n' % vocab_word.size()

    #################
    # Preprocessing #
    #################
    # samples: (word_ids, tag_ids, prd_indices, contexts)
    # word_ids: 1D: n_sents, 2D: n_words
    # tag_ids: 1D: n_sents, 2D: n_prds, 3D: n_words
    # prd_indices: 1D: n_sents, 2D: n_prds
    # contexts: 1D: n_sents, 2D: n_prds, 3D: n_words, 4D: window + 2
    # vocab_tags: {NA(Not-Arg):0, Ga:1, O:2, Ni:3, V:4}

    tr_pre_samples = get_samples(corpus=train_corpus, vocab_word=vocab_word, vocab_label=vocab_label,
                                 window=argv.window)
    tr_samples, tr_batch_index = theano_format(tr_pre_samples)
    print '\nTRAIN',
    sample_statistics(tr_pre_samples[1], vocab_label)
    n_train_batches = len(tr_batch_index)
    print '\tTrain Mini-Batches: %d\n' % n_train_batches

    if argv.dev_data:
        dev_pre_samples = get_samples(corpus=dev_corpus, vocab_word=vocab_word, vocab_label=vocab_label,
                                      window=argv.window)
        dev_samples, dev_batch_index = theano_format(dev_pre_samples)
        print '\nDEV',
        sample_statistics(dev_pre_samples[1], vocab_label)
        n_dev_batches = len(dev_batch_index)
        print '\tDev Mini-Batches: %d\n' % n_dev_batches

    if argv.test_data:
        test_pre_samples = get_samples(corpus=test_corpus, vocab_word=vocab_word, vocab_label=vocab_label,
                                       window=argv.window)
        test_samples, test_batch_index = theano_format(test_pre_samples)
        print '\nTEST',
        sample_statistics(test_pre_samples[1], vocab_label)
        n_test_batches = len(test_batch_index)
        print '\tTest Mini-Batches: %d\n' % n_test_batches

    if argv.check:
        eval = Eval()
#        eval.check_spans(tr_pre_samples, vocab_word, vocab_label)
        eval.check_num_spans(train_corpus, tr_pre_samples, vocab_label)
        exit()
        check_samples(tr_pre_samples, vocab_word, vocab_label)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '\n\nBuilding a model...'
    model = set_model(argv=argv, emb=emb, vocab_word=vocab_word, vocab_label=vocab_label)
    train_f = set_train_f(model, tr_samples, argv.dropout)
    if argv.dev_data:
        dev_f = set_pred_f(model, dev_samples)
    if argv.test_data:
        test_f = set_pred_f(model, test_samples)

    ###############
    # TRAIN MODEL #
    ###############
    print '\nTRAINING START\n'

    tr_indices = range(n_train_batches)
    if argv.dev_data:
        dev_indices = range(n_dev_batches)
    if argv.test_data:
        test_indices = range(n_test_batches)

    best_dev_f1 = -1.
    best_test_f1 = -1.

    for epoch in xrange(argv.epoch):
        train_eval = Eval()
        print '\nEpoch: %d' % (epoch + 1)
        print '  TRAIN\n\t',

        np.random.shuffle(tr_indices)
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            batch_range = tr_batch_index[b_index]
            result_sys, result_gold, nll = train_f(index=b_index, bos=batch_range[0], eos=batch_range[1])

            train_eval.update_results(result_sys, result_gold)
            train_eval.nll += nll

        print '\n\tTime: %f' % (time.time() - start)
        train_eval.show_results()

        """ Validating """
        update = False
        if argv.dev_data:
            print '\n  DEV\n\t',
            dev_f1 = predict(dev_f, dev_batch_index, dev_indices)
            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                update = True

        if argv.test_data:
            print '\n  TEST\n\t',
            test_f1 = predict(test_f, test_batch_index, test_indices)
            if update:
                best_test_f1 = test_f1

        print '\n\tBEST DEV F: %f  TEST F: %f' % (best_dev_f1, best_test_f1)


def predict(f, batch_index, indices):
    pred_eval = Eval()
    start = time.time()

    for index, b_index in enumerate(indices):
        if index != 0 and index % 1000 == 0:
            print index,
            sys.stdout.flush()

        batch_range = batch_index[b_index]
        result_sys, result_gold = f(index=b_index, bos=batch_range[0], eos=batch_range[1])
        pred_eval.update_results(result_sys, result_gold)

    print '\n\tTime: %f' % (time.time() - start)
    pred_eval.show_results()

    return pred_eval.all_f1


def main(argv):
    train(argv)
