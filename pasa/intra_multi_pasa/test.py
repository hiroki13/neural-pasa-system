import sys
import time

from nn_utils import io_utils
from nn_utils.io_utils import say
from preprocessor import get_samples, theano_format_online
from stats.stats import corpus_statistics, sample_statistics
from eval import Eval
from nn_utils.io_utils import load_data


def main(argv):
    say('\nLoading...\n\n')
    model = load_data(argv.load_params)

    if model.predict_all is None:
        model.set_predict_f()

    vocab_label = load_data(argv.label)
    print '\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i))

    vocab_word = load_data(argv.vocab)
    print '\nVocab: %d\tType: word\n' % vocab_word.size()

    ##############
    # Load files #
    ##############
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    if argv.dev_data:
        dev_corpus, _ = io_utils.load_ntc(path=argv.dev_data, data_size=argv.data_size, model='word')
        print '\nDEV',
        corpus_statistics(dev_corpus)

    if argv.test_data:
        test_corpus, _ = io_utils.load_ntc(path=argv.test_data, data_size=argv.data_size, model='word')
        print '\nTEST',
        corpus_statistics(test_corpus)

    if argv.dev_data:
        dev_pre_samples, dev_word_ids = get_samples(dev_corpus, vocab_word, vocab_label, argv.window)
        dev_samples = theano_format_online(dev_pre_samples)
        n_dev_batches = len(dev_samples[-1])
        print '\nDEV',
        sample_statistics(dev_pre_samples[1], vocab_label)
        print '\tDev Mini-Batches: %d\n' % n_dev_batches

    if argv.test_data:
        test_pre_samples, test_word_ids = get_samples(test_corpus, vocab_word, vocab_label, argv.window)
        test_samples = theano_format_online(test_pre_samples)
        n_test_batches = len(test_samples[-1])
        print '\nTEST',
        sample_statistics(test_pre_samples[1], vocab_label)
        print '\tTest Mini-Batches: %d\n' % n_test_batches

    print '\n\nBuilding a model...'

    if argv.dev_data:
        print '\n  DEV\n\t',
        dev_f1 = predict(model, dev_samples)
        say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

    if argv.test_data:
        print '\n  TEST\n\t',
        test_f1 = predict(model, test_samples)
        say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))


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

        results_sys, results_gold = model.predict_all(x, y, sent_len)
        pred_eval.update_results(results_sys, results_gold)

    print '\tTime: %f' % (time.time() - start)
    pred_eval.show_results()

    return pred_eval.all_f1
