import sys
import time
import numpy as np

import io_utils
from preprocessor import get_sample_info, theano_format, corpus_statistics, sample_statistics, check_samples
from model_builder import set_model, set_train_f, set_pred_f


def train(argv):
    print '\nSETTING UP A TRAINING SETTING\n'

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


def eval_args(batch_y_hat, batch_y):
    assert len(batch_y_hat) == len(batch_y)
    assert len(batch_y_hat[0]) == len(batch_y[0])

    crr = np.zeros(3, dtype='float32')
    ttl_p = np.zeros(3, dtype='float32')
    ttl_r = np.zeros(3, dtype='float32')

    for i in xrange(len(batch_y_hat)):
        sent_y_hat = batch_y_hat[i]
        sent_y = batch_y[i]
        for j in xrange(len(sent_y_hat)):
            y_hat = sent_y_hat[j]
            y = sent_y[j]

            if 0 < y_hat == y < 4:
                crr[y_hat-1] += 1
            if 0 < y_hat < 4:
                ttl_p[y_hat-1] += 1
            if 0 < y < 4:
                ttl_r[y-1] += 1

    return crr, ttl_p, ttl_r


def eval_char_args(batch_y_hat, batch_y):
    assert len(batch_y_hat) == len(batch_y)
    assert len(batch_y_hat[0]) == len(batch_y[0])
    crr = np.zeros(3, dtype='float32')
    ttl_p = np.zeros(3, dtype='float32')
    ttl_r = np.zeros(3, dtype='float32')

    for i in xrange(len(batch_y_hat)):
        y_spans = get_spans(batch_y[i])
        y_hat_spans = get_spans(batch_y_hat[i])

        for s1 in y_spans:
            span1 = s1[0]
            label1 = s1[1]

            for s2 in y_hat_spans:
                span2 = s2[0]
                label2 = s2[1]
                if span1 == span2:
                    if 1 <= label1 <= 2 and 1 <= label2 <= 2:
                        crr[0] += 1
                    elif 3 <= label1 <= 4 and 3 <= label2 <= 4:
                        crr[1] += 1
                    elif 5 <= label1 <= 6 and 5 <= label2 <= 6:
                        crr[2] += 1

                if 1 <= label2 <= 2:
                    ttl_p[0] += 1
                elif 3 <= label2 <= 4:
                    ttl_p[1] += 1
                elif 5 <= label2 <= 6:
                    ttl_p[2] += 1

            if 1 <= label1 <= 2:
                ttl_r[0] += 1
            elif 3 <= label1 <= 4:
                ttl_r[1] += 1
            elif 5 <= label1 <= 6:
                ttl_r[2] += 1

    return crr, ttl_p, ttl_r


def get_spans(y):
    spans = []

    for i, label in enumerate(y):
        if label < 1 or label > 6:
            continue

        if len(spans) == 0:
            spans.append(((i, i+1), label))
        else:
            prev = spans[-1]
            prev_span = prev[0]
            prev_label = prev[1]

            if prev_span[1] == i and (label == prev_label or (label == prev_label + 1 and label % 2 == 0)):
                spans.pop()
                spans.append(((prev_span[0], i+1), label))
            else:
                spans.append(((i, i+1), label))
    return spans


def check(argv):
    print '\nCHECKING THE DATASETS\n'
    corpus, vocab_word = io_utils.load_ntc(argv.train_data)
    true_corpus, vocab_word, prds = io_utils.load_converted_ntc(argv.true_data)

    print '\nCHECKED CORPUS',
    corpus_statistics(corpus)

    print '\nTRUE CORPUS',
    corpus_statistics(true_corpus)

    check_num_prds(corpus, prds)


def check_num_prds(corpus, prds):
    prd_count1 = 0
    prd_count2 = 0

    for doc, doc_prds in zip(corpus, prds):
        assert len(doc) == len(doc_prds), 'DOC: %d\t%d' % (len(doc), len(doc_prds))
        for sent, sent_prds in zip(doc, doc_prds):
            count = 0
            for w in sent:
                if w.is_prd:
                    count += 1
            if count != len(sent_prds):
                print 'PRDS: %d\t%d' % (count, len(sent_prds))
                for w in sent:
                    print '%s ' % w.form,
                    for e in w.pas_info:
                        print e,
                    print
            prd_count1 += count
            prd_count2 += len(sent_prds)
    print 'PRD: %d  %d' % (prd_count1, prd_count2)


def main(argv):
    if argv.true_data:
        check(argv)
    else:
        train(argv)

