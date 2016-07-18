import sys
import time
import numpy as np

from utils import io_utils
from preprocessor import get_sample_info, theano_format, corpus_statistics, sample_statistics, check_samples, get_inter_samples, theano_format_inter
from model_builder import set_model, set_train_f, set_pred_f
from eval import eval_args, eval_char_args


def train(argv):
    print '\nSETTING UP AN INTRA-SENTENTIAL TRAINING SETTING\n'

    emb = None

    """ Load files """
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    tr_corpus, vocab_word = io_utils.load_ntc(argv.train_data, argv.data_size, argv.vocab_size, argv.model)
    dev_corpus, vocab_word = io_utils.load_ntc(argv.dev_data, argv.data_size, argv.vocab_size, argv.model, vocab_word)
    test_corpus, vocab_word = io_utils.load_ntc(argv.test_data, argv.data_size, argv.vocab_size, argv.model, vocab_word)

    print '\nVocab: %d\tType: %s\n' % (vocab_word.size(), argv.model)
    print '\nTRAIN',
    corpus_statistics(tr_corpus)
    print '\nDEV',
    corpus_statistics(dev_corpus)
    print '\nTEST',
    corpus_statistics(test_corpus)

    """ Preprocessing """
    # samples: (word_ids, tag_ids, prd_indices, contexts)
    # word_ids: 1D: n_sents, 2D: n_words
    # tag_ids: 1D: n_sents, 2D: n_prds, 3D: n_words
    # prd_indices: 1D: n_sents, 2D: n_prds
    # contexts: 1D: n_sents, 2D: n_prds, 3D: n_words, 4D: window + 2
    # vocab_tags: {NA(Not-Arg):0, Ga:1, O:2, Ni:3, V:4}
    tr_pre_samples, vocab_label = get_sample_info(tr_corpus, vocab_word, argv.model, window=argv.window)
    dev_pre_samples, vocab_label = get_sample_info(dev_corpus, vocab_word, argv.model, vocab_label, argv.window)
    test_pre_samples, vocab_label = get_sample_info(test_corpus, vocab_word, argv.model, vocab_label, argv.window)
    print '\nLabel: %d\n' % vocab_label.size()

    print '\nTRAIN',
    sample_statistics(tr_pre_samples[1], vocab_label)
    print '\nDEV',
    sample_statistics(dev_pre_samples[1], vocab_label)
    print '\nTEST',
    sample_statistics(test_pre_samples[1], vocab_label)

    if argv.check:
        check_samples(tr_pre_samples, vocab_word, vocab_label)

    # dataset: (labels, contexts, sent_length)
    tr_samples, tr_batch_index = theano_format(tr_pre_samples)
    dev_samples, dev_batch_index = theano_format(dev_pre_samples)
    test_samples, test_batch_index = theano_format(test_pre_samples)
    n_tr_batch = len(tr_batch_index)
    n_dev_batch = len(dev_batch_index)
    n_test_batch = len(test_batch_index)
    print '\nBatches: Train: %d  Dev: %d  Test: %d\n' % (n_tr_batch, n_dev_batch, n_test_batch)

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

    print '\nTRAINING START\n'

    tr_indices = range(n_tr_batch)
    dev_indices = range(n_dev_batch)
    test_indices = range(n_test_batch)
    best_dev_f1 = -1.
    best_test_f1 = -1.

    for epoch in xrange(argv.epoch):
        print '\nEpoch: %d' % (epoch + 1)
        print '  TRAIN\n\t',

        np.random.shuffle(tr_indices)
        ttl_p = np.zeros(3, dtype='float32')
        ttl_r = np.zeros(3, dtype='float32')
        crr = np.zeros(3, dtype='float32')
        ttl_nll = 0.
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            batch_range = tr_batch_index[b_index]
            pred, nll, y = train_f(index=b_index, bos=batch_range[0], eos=batch_range[1])
            if model == 'word':
                crr_i, ttl_p_i, ttl_r_i = eval_args(pred, y)
            else:
                crr_i, ttl_p_i, ttl_r_i = eval_char_args(pred, y)
            crr += crr_i
            ttl_p += ttl_p_i
            ttl_r += ttl_r_i
            ttl_nll += nll

        end = time.time()
        print '\n\tTime: %f  NLL: %f' % ((end - start), ttl_nll)

        """ Evaluation per case """
        for c in xrange(3):
            print '\n\tCASE: %d' % c
            precision = crr[c] / ttl_p[c]
            recall = crr[c] / ttl_r[c]
            f = 2 * precision * recall / (precision + recall)
            print '\tCRR: %d  TTL P: %d  TTL R: %d' % (crr[c], ttl_p[c], ttl_r[c])
            print '\tF: %f  P: %f  R: %f' % (f, precision, recall)

        """ Evaluation for all the cases """
        t_crr = np.sum(crr)
        t_ttl_p = np.sum(ttl_p)
        t_ttl_r = np.sum(ttl_r)

        precision = t_crr / t_ttl_p
        recall = t_crr / t_ttl_r
        f = 2 * precision * recall / (precision + recall)
        print '\n\tTOTAL'
        print '\tCRR: %d  TTL P: %d  TTL R: %d' % (t_crr, t_ttl_p, t_ttl_r)
        print '\tF: %f  P: %f  R: %f' % (f, precision, recall)

        """ Validating """
        update = False
        if argv.dev_data:
            print '\n  DEV\n\t',
            dev_f1 = predict(dev_f, dev_batch_index, dev_indices, argv.model)
            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                update = True

        if argv.test_data:
            print '\n  TEST\n\t',
            test_f1 = predict(test_f, test_batch_index, test_indices, argv.model)
            if update:
                best_test_f1 = test_f1

        print '\n\tBEST DEV F: %f  TEST F: %f' % (best_dev_f1, best_test_f1)


def predict(f, batch_index, indices, model='word'):
    ttl_p = np.zeros(3, dtype='float32')
    ttl_r = np.zeros(3, dtype='float32')
    crr = np.zeros(3, dtype='float32')
    start = time.time()

    for index, b_index in enumerate(indices):
        if index != 0 and index % 1000 == 0:
            print index,
            sys.stdout.flush()

        batch_range = batch_index[b_index]
        pred, y = f(index=b_index, bos=batch_range[0], eos=batch_range[1])
        if model == 'word':
            crr_i, ttl_p_i, ttl_r_i = eval_args(pred, y)
        else:
            crr_i, ttl_p_i, ttl_r_i = eval_char_args(pred, y)
        crr += crr_i
        ttl_p += ttl_p_i
        ttl_r += ttl_r_i

    end = time.time()
    print '\n\tTime: %f' % (end - start)

    """ Evaluation per case """
    for c in xrange(3):
        print '\n\tCASE: %d' % c
        precision = crr[c] / ttl_p[c]
        recall = crr[c] / ttl_r[c]
        f = 2 * precision * recall / (precision + recall)
        print '\tCRR: %d  TTL P: %d  TTL R: %d' % (crr[c], ttl_p[c], ttl_r[c])
        print '\tF: %f  P: %f  R: %f' % (f, precision, recall)

    """ Evaluation for all the cases """
    t_crr = np.sum(crr)
    t_ttl_p = np.sum(ttl_p)
    t_ttl_r = np.sum(ttl_r)

    precision = t_crr / t_ttl_p
    recall = t_crr / t_ttl_r
    f = 2 * precision * recall / (precision + recall)
    print '\n\tTOTAL'
    print '\tCRR: %d  TTL P: %d  TTL R: %d' % (t_crr, t_ttl_p, t_ttl_r)
    print '\tF: %f  P: %f  R: %f' % (f, precision, recall)

    return f


def main(argv):
    train(argv)
