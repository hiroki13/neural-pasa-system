from collections import defaultdict

from ..utils.io_utils import say
from ..utils.preprocessor import Preprocessor
from ..ling.vocab import UNK
from ..utils.stats import corpus_statistics


class TextGenerator(Preprocessor):

    def __init__(self, argv):
        super(TextGenerator, self).__init__(argv)

    def load_corpus(self):
        # corpus: 1D: n_sents, 2D: n_words, 3D: Word()
        return self.corpus_loader.load_corpus(path=self.argv.data)

    @staticmethod
    def get_word_freqs(corpus):
        word_freqs = defaultdict(int)
        for doc in corpus:
            for sent in doc:
                for w in sent:
                    word_freqs[w.form] += 1
        say('\nVocab: %d\n' % len(word_freqs))
        return word_freqs

    def set_unknown(self, word_freqs):
        count = 0
        for k, v in sorted(word_freqs.items(), key=lambda x: -x[1]):
            if v > self.argv.vocab_cut_off:
                continue
            count += v
        word_freqs[UNK] = count
        return word_freqs

    def output_text(self, corpus, word_freqs):
        fn = self.argv.out_data
        with open(fn, 'w') as fout:
            for doc in corpus:
                for sent in doc:
                    text = ''
                    for w in sent:
                        w_freq = word_freqs[w.form]
                        if w_freq > self.argv.vocab_cut_off:
                            word = w.form
                        else:
                            word = UNK
                        text += '%s ' % word
                    print >> fout, text.encode('utf-8')

    def output_vocab_list(self, word_freqs):
        fn = self.argv.out_vocab
        with open(fn, 'w') as fout:
            for k, v in sorted(word_freqs.items(), key=lambda x: -x[1]):
                if v <= self.argv.vocab_cut_off:
                    break
                text = '%s %d' % (k, v)
                print >> fout, text.encode('utf-8')

    @staticmethod
    def show_corpus_stats(corpus):
        corpus_statistics(corpus)


def main(argv):
    say('\n\nThe Text Generator START\n')

    text_generator = TextGenerator(argv)

    text_generator.select_corpus_loader()
    corpus = text_generator.load_corpus()
    text_generator.show_corpus_stats(corpus)

    word_freqs = text_generator.get_word_freqs(corpus)
    word_freqs = text_generator.set_unknown(word_freqs)
    text_generator.output_text(corpus, word_freqs)
    text_generator.output_vocab_list(word_freqs)


