# -*- coding: utf-8 -*-

import gzip
import cPickle
from collections import defaultdict

from ling.word import Word


def load_ntc(path, data_size=1000000, model='word', word_freqs=defaultdict(int)):
    corpus = []
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
                w.sent_index = len(doc)
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

    return corpus, word_freqs


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn[-7:] != '.pkl.gz':
        fn += '.pkl.gz'

    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)
