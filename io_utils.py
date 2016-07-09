# -*- coding: utf-8 -*-

import gzip
import cPickle
from collections import defaultdict

from vocab import Vocab
from word import Word

PAD = u'<PAD>'
UNK = u'<UNK>'
MARK = u'<MARK>'
NMARK = u'<NMARK>'


def load_ntc(path, data_size=1000000, vocab_threshold=0, model='word', vocab_word=Vocab()):
    corpus = []
    word_freqs = defaultdict(int)

    if vocab_word.size() == 0:
        vocab_word.add_word(PAD)
        for i in xrange(11):
            vocab_word.add_word('<DISTANCE-%d>' % i)
#        vocab_word.add_word(NMARK)
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
                pass
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
                doc.append(sent)
                sent = []
                w_index = 0
            else:
                w = Word(w_index, elem)
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
        if freq == vocab_threshold:
            break
        vocab_word.add_word(w)

    return corpus, vocab_word


def load_converted_ntc(path, vocab_word=Vocab()):
    corpus = []
    word_freqs = defaultdict(int)

    if vocab_word.size() == 0:
        vocab_word.add_word(PAD)
        vocab_word.add_word(UNK)

    with open(path) as freq:
        prev_doc_id = None
        doc = []
        sent = []
        doc_prds = []
        sent_prds = []
        prds = []
        w_index = 0

        for line in freq:
            elem = line.rstrip().split()

            if line.startswith('*'):
                if elem[-1] == 'PRED':
                    sent_prds.append(elem)
            elif line.startswith('#'):  # Doc starts
                doc_id = line.split()[1]
                if prev_doc_id and prev_doc_id != doc_id:
                    prev_doc_id = doc_id
                    corpus.append(doc)
                    prds.append(doc_prds)
                    doc = []
                    doc_prds = []
                elif prev_doc_id is None:
                    prev_doc_id = doc_id
            elif line.startswith('EOS'):  # Sent ends
                doc.append(sent)
                doc_prds.append(sent_prds)
                sent = []
                sent_prds = []
                w_index = 0
            else:
                w = Word(w_index, elem)
                word_freqs[w.form] += 1
                sent.append(w)
                w_index += 1

        if doc:
            corpus.append(doc)
            prds.append(doc_prds)

    for w, freq in sorted(word_freqs.items(), key=lambda (k, v): -v):
        if freq == 1:
            break
        vocab_word.add_word(w)

    assert len(corpus) == len(prds)
    return corpus, vocab_word, prds


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn[-7:] != '.pkl.gz':
        fn += '.pkl.gz'

    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)
