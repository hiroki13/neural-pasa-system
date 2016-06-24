# -*- coding: utf-8 -*-

import gzip
import cPickle
from collections import defaultdict

from vocab import Vocab
from word import Word

PAD = u'<PAD>'
UNK = u'<UNK>'


def load_ntc(path):
    corpus = []
    word_freqs = defaultdict(int)
    vocab_word = Vocab()
    vocab_word.add_word(PAD)
    vocab_word.add_word(UNK)

    with open(path) as f:
        prev_doc_id = None
        doc = []
        sent = []
        w_index = 0

        for line in f:
            elem = line.rstrip().split()

            if line.startswith('*'):
                pass
            elif line.startswith('#'):  # Doc starts
                doc_id = line.split()[1].split(':')[1].split('-')[0]
                if prev_doc_id and prev_doc_id != doc_id:
                    prev_doc_id = doc_id
                    corpus.append(doc)
                    doc = []
                elif prev_doc_id is None:
                    prev_doc_id = doc_id
            elif line.startswith('EOS'):  # Sent ends
                doc.append(sent)
                sent = []
                w_index = 0
            else:
                w = Word(w_index, elem)
                word_freqs[w.form] += 1
                sent.append(w)
                w_index += 1

        if doc:
            corpus.append(doc)

    for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
        vocab_word.add_word(w)

    return corpus, vocab_word


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn[-7:] != '.pkl.gz':
        fn += '.pkl.gz'

    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)
