import sys
import time
import math

import theano
import numpy as np

from utils import io_utils
from utils.io_utils import say, dump_data
from ling.vocab import Vocab
from preprocessor import get_samples, get_shared_samples
from stats.stats import corpus_statistics, sample_statistics, check_samples
from decoder import Decoder
from eval import Eval


def train(argv):
    say('\n\nSETTING UP AN INTRA-SENTENTIAL TRAINING SETTING\n')

    emb = None
    mp = True if argv.attention else False

    ##############
    # Set labels #
    ##############
    vocab_label = Vocab()
    vocab_label.set_pas_labels()
    if argv.save:
        dump_data(vocab_label, 'vocab_label')
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
    if argv.save:
        dump_data(vocab_word, 'vocab_word.cut-%d' % argv.vocab_cut_off)
    print '\nVocab: %d\tType: word\n' % vocab_word.size()

    #################
    # Preprocessing #
    #################
    # samples: 1D: n_sents; Sample
    train_samples = get_samples(train_corpus, vocab_word, vocab_label, argv.window)
    train_sample_shared, train_batch_index = get_shared_samples(train_samples, batch_size=argv.batch_size, mp=mp)
    print '\nTRAIN',
    sample_statistics(train_samples, vocab_label)
    n_train_batches = len(train_batch_index)
    print '\tTrain Mini-Batches: %d\n' % n_train_batches

    if argv.dev_data:
        dev_samples = get_samples(dev_corpus, vocab_word, vocab_label, argv.window, test=True)
        n_dev_batches = len(dev_samples)
        print '\nDEV',
        sample_statistics(dev_samples, vocab_label)
        print '\tDev Mini-Batches: %d\n' % n_dev_batches

    if argv.test_data:
        test_samples = get_samples(test_corpus, vocab_word, vocab_label, argv.window, test=True)
        n_test_batches = len(test_samples)
        print '\nTEST',
        sample_statistics(test_samples, vocab_label)
        print '\tTest Mini-Batches: %d\n' % n_test_batches

    if argv.check:
        check_samples(train_samples, vocab_word, vocab_label)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '\n\nBuilding a model...'
    decoder = Decoder(argv=argv, emb=emb, vocab_word=vocab_word, vocab_label=vocab_label)
    decoder.set_model()

    decoder.set_train_f(train_sample_shared)
    if argv.dev_data or argv.test_data:
        decoder.set_predict_f()

    ###############
    # TRAIN MODEL #
    ###############
    print '\nTRAINING START\n'

    tr_indices = range(n_train_batches)

    f1_history = {}
    best_dev_f1 = -1.

    for epoch in xrange(argv.epoch):
        train_eval = Eval()
        dropout_p = np.float32(argv.dropout).astype(theano.config.floatX)
        decoder.model.dropout.set_value(dropout_p)

        print '\nEpoch: %d' % (epoch + 1)
        print '  TRAIN\n\t',

        np.random.shuffle(tr_indices)
        start = time.time()

        for index, b_index in enumerate(tr_indices):
            if index != 0 and index % 1000 == 0:
                print index,
                sys.stdout.flush()

            batch_range = train_batch_index[b_index]
            result_sys, result_gold, nll = decoder.train(index=b_index, bos=batch_range[0], eos=batch_range[1])

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
            dev_f1 = decoder.predict_all(dev_samples)
            if best_dev_f1 < dev_f1:
                best_dev_f1 = dev_f1
                f1_history[epoch+1] = [best_dev_f1]
                update = True

                if argv.save:
                    decoder.save_model('model.intra.layer-%d.window-%d.reg-%f' % (argv.layer, argv.window, argv.reg))
                    decoder.output_results('result.dev.intra.layer-%d.window-%d.reg-%f.txt' %
                                           (argv.layer, argv.window, argv.reg),
                                           dev_samples)

        if argv.test_data:
            print '\n  TEST\n\t',
            test_f1 = decoder.predict_all(test_samples)
            if update:
                decoder.output_results('result.test.intra.layer-%d.window-%d.reg-%f.txt' %
                                       (argv.layer, argv.window, argv.reg),
                                       test_samples)
                if epoch+1 in f1_history:
                    f1_history[epoch+1].append(test_f1)
                else:
                    f1_history[epoch+1] = [test_f1]

        say('\n\n\tF1 HISTORY')
        for k, v in sorted(f1_history.items()):
            if len(v) == 2:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}\tBEST TEST F:{:.2%}'.format(k, v[0], v[1]))
            else:
                say('\n\tEPOCH-{:d}  \tBEST DEV F:{:.2%}'.format(k, v[0]))
        say('\n\n')


def main(argv):
    train(argv)
