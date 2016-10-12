from ..utils.io_utils import say
from ..utils.preprocessor import Experimenter
from ..model.model_api import ModelAPI


def main(argv):
    say('\n\nSETTING UP AN INTRA-SENTENTIAL PASA TRAINING SETTING\n')

    experimenter = Experimenter(argv)

    experimenter.select_corpus_loader()
    corpus_set = experimenter.load_corpus_set()
    experimenter.show_corpus_stats(corpus_set)

    vocab_label = experimenter.create_vocab_label()
    vocab_word = experimenter.create_vocab_word(corpus_set[0])

    experimenter.select_preprocessor(vocab_word, vocab_label)
    sample_set = experimenter.create_sample_set(corpus_set)
    train_samples, dev_samples, test_samples = sample_set
    train_sample_shared, train_batch_index = experimenter.create_shared_samples(train_samples)
    experimenter.show_sample_stats(sample_set, vocab_label)

    model_api = ModelAPI(argv=argv, emb=None, vocab_word=vocab_word, vocab_label=vocab_label)
    model_api.compile(train_sample_shared)
    model_api.train_all(argv, train_batch_index, dev_samples, test_samples)
