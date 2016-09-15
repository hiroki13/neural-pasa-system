from utils import io_utils
from utils.io_utils import say, load_data
from preprocessor import get_samples
from stats.stats import corpus_statistics, sample_statistics
from decoder import Decoder


def main(argv):
    emb = None

    say('\nLoading...\n\n')

    ####################
    # Load vocab files #
    ####################
    vocab_label = load_data(argv.label)
    say('\nTARGET LABELS: %d\t%s\n' % (vocab_label.size(), str(vocab_label.w2i)))

    vocab_word = load_data(argv.vocab)
    say('\nVocab: %d\tType: word\n' % vocab_word.size())

    ##################
    # Load a decoder #
    ##################
    decoder = Decoder(argv, emb, vocab_word, vocab_label)
    decoder.load_model(argv)
    dec_argv = decoder.argv

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

    #################
    # Preprocessing #
    #################
    if argv.dev_data:
        dev_samples = get_samples(dev_corpus, vocab_word, vocab_label, dec_argv.window)
        n_dev_batches = len(dev_samples)
        print '\nDEV',
        sample_statistics(dev_samples, vocab_label)
        print '\tDev Mini-Batches: %d\n' % n_dev_batches

    if argv.test_data:
        test_samples = get_samples(test_corpus, vocab_word, vocab_label, dec_argv.window)
        n_test_batches = len(test_samples)
        print '\nTEST',
        sample_statistics(test_samples, vocab_label)
        print '\tTest Mini-Batches: %d\n' % n_test_batches

    ###############
    # Set a model #
    ###############
    say('\n\nBuilding a model...')
    decoder.set_predict_f()

    ###########
    # Predict #
    ###########
    if argv.dev_data:
        print '\n  DEV\n\t',
        dev_f1 = decoder.predict_all(dev_samples)
        say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))

    if argv.test_data:
        print '\n  TEST\n\t',
        test_f1 = decoder.predict_all(test_samples)
        say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))

