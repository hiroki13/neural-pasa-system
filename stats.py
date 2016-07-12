from collections import defaultdict
import numpy as np

from vocab import Vocab
from word import Word
from preprocessor import get_sample_info, corpus_statistics, sample_statistics, check_samples


PAD = u'<PAD>'
UNK = u'<UNK>'
MARK = u'<MARK>'
NMARK = u'<NMARK>'


def load_ntc(path, data_size=1000000, vocab_size=0, model='word', vocab_word=Vocab()):
    corpus = []
    word_freqs = defaultdict(int)

    if vocab_word.size() == 0:
        vocab_word.add_word(PAD)
        vocab_word.add_word(MARK)
        vocab_word.add_word(NMARK)
        vocab_word.add_word(UNK)

    flag = True
    with open(path) as f:
        prev_doc_id = None
        doc = []
        sent = []
        w_index = 0

        for line in f:
            elem = line.rstrip().split()

            if line.startswith('*'):
                chunk_index = int(elem[1])
                chunk_head = int(elem[2][:-1])
            elif line.startswith('#'):  # Doc starts
                doc_id = line.split()[1].split(':')[1].split('-')[0]
                if prev_doc_id and prev_doc_id != doc_id:
                    prev_doc_id = doc_id
                    corpus.append(doc)
                    doc = []

                    if len(corpus) == data_size:
                        flag = False
                        break
                elif prev_doc_id is None:
                    prev_doc_id = doc_id
            elif line.startswith('EOS'):  # Sent ends
                for w in sent:
                    w.set_cases(sent, doc)
                doc.append(sent)
                sent = []
                w_index = 0
            else:
                w = Word(w_index, elem)
                w.chunk_index = chunk_index
                w.chunk_head = chunk_head
                if model == 'word':
                    word_freqs[w.form] += 1
                else:
                    for c in w.chars:
                        word_freqs[c] += 1
                sent.append(w)
                w_index += 1

        if doc and flag:
            corpus.append(doc)

    for w, freq in sorted(word_freqs.items(), key=lambda (k, v): -v):
        if freq == vocab_size:
            break
        vocab_word.add_word(w)

    return corpus, vocab_word


def show_case_dist(corpus):
    case_types = np.zeros((3, 5))
    n_prds = 0

    for doc in corpus:
        for sent in doc:
            for w in sent:
                if w.is_prd is False:
                    continue
                flag = False
#                print '%s\t%d\t%d\t%d\n' % (w.form, w.id, w.chunk_index, w.chunk_head),
                w_case_arg_ids = w.case_arg_ids
                w_case_types = w.case_types
                for i, (a, t) in enumerate(zip(w_case_arg_ids, w_case_types)):
                    if t > -1:
                        case_types[i][t] += 1
                        if 0 < t < 3:
                            flag = True
                if flag:
                    n_prds += 1
    print case_types
    print 'PRDS: %d' % n_prds


def show_stats(argv):
    print '\nDATASET STATISTICS\n'

    """ Load files """
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    tr_corpus, vocab_word = load_ntc(argv.train_data, argv.data_size, argv.vocab_size, argv.model)

    print '\nVocab: %d\tType: %s\n' % (vocab_word.size(), argv.model)
    print '\nTRAIN',
    corpus_statistics(tr_corpus)

    """
    DEV
    Ga: dep: 7436, ZERO: 2665
    O : DEP: 5083, ZERO: 418
    Ni: DEP: 1612, ZERO: 137
    TEST
    Ga: dep: 14074, ZERO: 4942
    O : DEP: 9485,  ZERO: 830
    Ni: DEP: 2517,  ZERO: 251
    """
    show_case_dist(tr_corpus)

    """ Preprocessing """
    # samples: (word_ids, tag_ids, prd_indices, contexts)
    # word_ids: 1D: n_sents, 2D: n_words
    # tag_ids: 1D: n_sents, 2D: n_prds, 3D: n_words
    # prd_indices: 1D: n_sents, 2D: n_prds
    # contexts: 1D: n_sents, 2D: n_prds, 3D: n_words, 4D: window + 2
    # vocab_tags: {NA(Not-Arg):0, Ga:1, O:2, Ni:3, V:4}
    tr_pre_samples, vocab_label = get_sample_info(tr_corpus, vocab_word, argv.model, window=argv.window)
    print '\nLabel: %d\n' % vocab_label.size()

    print '\nTRAIN',
    sample_statistics(tr_pre_samples[1], vocab_label)


def main(argv):
    show_stats(argv)

