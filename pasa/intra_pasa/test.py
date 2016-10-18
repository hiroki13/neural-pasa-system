from ..utils.io_utils import say
from ..utils.preprocessor import Preprocessor
from ..model.model_api import ModelAPI


def main(argv):
    say('\n\nSETTING UP AN INTRA-SENTENTIAL PASA TEST SETTING\n')

    experimenter = Preprocessor(argv)

    config = experimenter.load_config()
    vocab_label = experimenter.load_labels()
    vocab_word = experimenter.load_words()

    experimenter.select_corpus_loader()
    corpus_set = experimenter.load_corpus_set()
    experimenter.show_corpus_stats(corpus_set)

    experimenter.select_preprocessor(vocab_word, vocab_label)
    sample_set = experimenter.create_sample_set(corpus_set)
    train_samples, dev_samples, test_samples = sample_set
    experimenter.show_sample_stats(sample_set, vocab_label)

    model_api = ModelAPI(argv=argv, emb=None, vocab_word=vocab_word, vocab_label=vocab_label)
    model_api.set_model()
    model_api.compile_model()
    model_api.load_params(argv.load_params)
    model_api.set_predict_f()

    ###########
    # Predict #
    ###########
    if argv.dev_data:
        print '\n  DEV\n\t',
        dev_f1 = model_api.predict_all(dev_samples)
        say('\n\n\tDEV F:{:.2%}\n'.format(dev_f1))
        model_api.output_results('result.dev.intra.layers-%d.window-%d.reg-%f.txt' %
                                 (config.layers, config.window, config.reg), dev_samples)

    if argv.test_data:
        print '\n  TEST\n\t',
        test_f1 = model_api.predict_all(test_samples)
        say('\n\n\tBEST TEST F:{:.2%}\n'.format(test_f1))
        model_api.output_results('result.test.intra.layers-%d.window-%d.reg-%f.txt' %
                                 (config.layers, config.window, config.reg), test_samples)

