import sys
import time
import numpy as np

import io_utils
from preprocessor import get_sample_info, theano_format, corpus_statistics
from model_builder import set_model, set_train_f, set_pred_f


def train(argv):
    print 'SETTING UP A TRAINING SETTING'

    emb = None

    """ Load files """
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    tr_corpus, vocab_word = io_utils.load_ntc(argv.train_data)
    dev_corpus, vocab_word = io_utils.load_ntc(argv.dev_data, vocab_word)
    test_corpus, vocab_word = io_utils.load_ntc(argv.test_data, vocab_word)

    print '\nTRAIN',
    corpus_statistics(tr_corpus)
    print '\nDEV',
    corpus_statistics(dev_corpus)
    print '\nTEST',
    corpus_statistics(test_corpus)

    """ Preprocessing """
    # samples: 1D: n_sents, 2D: [word_ids, tag_ids, prd_indices, contexts]
    # vocab_tags: {PAD:0, Ga:1, O:2, Ni:3, V:4}
    tr_dataset, vocab_label = get_sample_info(tr_corpus, vocab_word)
    dev_dataset, vocab_label = get_sample_info(dev_corpus, vocab_word, vocab_label)
    test_dataset, vocab_label = get_sample_info(test_corpus, vocab_word, vocab_label)

    # dataset: (labels, contexts, sent_length)
    tr_samples, tr_batch_index = theano_format(tr_dataset)
    dev_samples, dev_batch_index = theano_format(dev_dataset)
    test_samples, test_batch_index = theano_format(test_dataset)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    """ Set a model """
    print '\n\nBuilding a model...'
    model = set_model(argv=argv, emb=emb, vocab_word=vocab_word, vocab_label=vocab_label)
    train_f = set_train_f(model, tr_samples)
    dev_f = set_pred_f(model, dev_samples)
    test_f = set_pred_f(model, test_samples)

    ###############
    # TRAIN MODEL #
    ###############

    print '\nTraining start\n'
    tr_indices = range(len(tr_batch_index))
    dev_indices = range(len(dev_batch_index))
    test_indices = range(len(test_batch_index))

    for epoch in xrange(argv.epoch):
        print '\nEpoch: %d' % (epoch + 1)
        print '\tTRAIN\n\t',

        np.random.shuffle(tr_indices)
        best_dev_f1 = -1.
        best_test_f1 = -1.
        ttl_p = 0.
        ttl_r = 0.
        crr = 0.
        ttl = 0.
        ttl_nll = 0.
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 100 == 0:
                print index,
                sys.stdout.flush()

            batch_range = tr_batch_index[b_index]
            pred, nll, y = train_f(index=b_index, bos=batch_range[0], eos=batch_range[1])
            crr_i, ttl_p_i, ttl_r_i, ttl_i = eval_args(pred, y)
            crr += crr_i
            ttl_p += ttl_p_i
            ttl_r += ttl_r_i
            ttl += ttl_i
            ttl_nll += nll

        end = time.time()
        print '\n\tTime: %f  NLL: %f' % ((end - start), ttl_nll)

        precision = crr / ttl_p
        recall = crr / ttl_r
        f = 2 * precision * recall / (precision + recall)
        print '\tCRR: %d  TTL P: %d  TTL R: %d  TTL: %d' % (crr, ttl_p, ttl_r, ttl)
        print '\tF: %f  P: %f  R: %f' % (f, precision, recall)

        update = False
        if argv.dev_data:
            print '\n\tDEV\n\t',
            dev_f1 = predict(dev_f, dev_batch_index, dev_indices)
            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                update = True

        if argv.test_data:
            print '\n\tTEST\n\t',
            test_f1 = predict(test_f, test_batch_index, test_indices)
            if update:
                best_test_f1 = test_f1

        print '\n\tBEST DEV F: %f  TEST F: %f' % (best_dev_f1, best_test_f1)


def predict(f, batch_index, indices):
    ttl_p = 0.
    ttl_r = 0.
    crr = 0.
    ttl = 0.
    start = time.time()

    for index, b_index in enumerate(indices):
        if index != 0 and index % 100 == 0:
            print index,
            sys.stdout.flush()

        batch_range = batch_index[b_index]
        pred, y = f(index=b_index, bos=batch_range[0], eos=batch_range[1])
        crr_i, ttl_p_i, ttl_r_i, ttl_i = eval_args(pred, y)
        crr += crr_i
        ttl_p += ttl_p_i
        ttl_r += ttl_r_i
        ttl += ttl_i

    end = time.time()
    print '\n\tTime: %f' % (end - start)

    precision = crr / ttl_p
    recall = crr / ttl_r
    f = 2 * precision * recall / (precision + recall)
    print '\tCRR: %d  TTL P: %d  TTL R: %d  TTL: %d' % (crr, ttl_p, ttl_r, ttl)
    print '\tF: %f  P: %f  R: %f' % (f, precision, recall)

    return f


def eval_args(batch_y_hat, batch_y):
    assert len(batch_y_hat) == len(batch_y)
    assert len(batch_y_hat[0]) == len(batch_y[0])

    crr = 0.
    ttl_p = 0.
    ttl_r = 0.
    ttl = 0.

    for i in xrange(len(batch_y_hat)):
        sent_y_hat = batch_y_hat[i]
        sent_y = batch_y[i]
        for j in xrange(len(sent_y_hat)):
            y_hat = sent_y_hat[j]
            y = sent_y[j]

            if 0 < y_hat == y < 4:
                crr += 1
            if 0 < y_hat < 4:
                ttl_p += 1
            if 0 < y < 4:
                ttl_r += 1
            ttl += 1

    return crr, ttl_p, ttl_r, ttl

