import sys
import os
import shutil
import gzip
import cPickle
from abc import ABCMeta, abstractmethod

import numpy as np
import theano

from ..ling.vocab import Vocab, PAD, UNK
from ..ling.word import Word, ConllWord
from ..ling.sent import Sentence


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


class CorpusLoader(object):
    __metaclass__ = ABCMeta

    def __init__(self, min_unit, data_size):
        self.min_unit = min_unit
        self.data_size = data_size

    @abstractmethod
    def load_corpus(self, path):
        raise NotImplementedError


class NTCLoader(CorpusLoader):

    def load_corpus(self, path):
        if path is None:
            return None

        BOD = '#'
        BOC = '*'
        EOS = 'EOS'

        corpus = []
        with open(path) as f:
            prev_doc_id = None
            doc = []
            sent = []
            chunk_index = None
            chunk_head = None

            for line in f:
                elem = line.rstrip().split()

                if line.startswith(BOD):
                    doc_id = self._get_doc_id(elem)

                    if prev_doc_id and prev_doc_id != doc_id:
                        prev_doc_id = doc_id
                        corpus.append(doc)
                        doc = []
                    elif prev_doc_id is None:
                        prev_doc_id = doc_id
                elif line.startswith(BOC):
                    chunk_index, chunk_head = self._get_chunk_info(elem)
                elif line.startswith(EOS):
                    for w in sent:
                        w.set_cases(sent)
                    doc.append(sent)
                    sent = []
                else:
                    word = self._get_word(w_index=len(sent),
                                          chunk_index=chunk_index,
                                          chunk_head=chunk_head,
                                          sent_index=len(doc),
                                          elem=elem)
                    sent.append(word)

                if len(corpus) == self.data_size:
                    break
            else:
                if doc:
                    corpus.append(doc)

        return corpus

    @staticmethod
    def _get_doc_id(elem):
        return elem[1].split(':')[1].split('-')[0]

    @staticmethod
    def _get_chunk_info(elem):
        return int(elem[1]), int(elem[2][:-1])

    @staticmethod
    def _get_word(w_index, chunk_index, chunk_head, sent_index, elem):
        w = Word(w_index, elem)
        w.sent_index = sent_index
        w.chunk_index = chunk_index
        w.chunk_head = chunk_head
        return w


class CONLLLoader(CorpusLoader):

    def load_corpus(self, path):
        PRD = '#'
        RESULT = '*'
        corpus = []
        with open(path) as f:
            sent = Sentence()
            for line in f:
                line = line.rstrip()
                elem = line.split('\t')

                if len(line) == 0:
                    corpus.append(sent)
                    sent = Sentence()
                elif elem[0] == PRD:
                    sent.set_prd(elem)
                elif elem[0] == RESULT:
                    sent.set_args(elem)
                else:
                    sent.words.append(ConllWord(elem))
        return corpus


def load_init_emb(fn, dim_emb):
    """
    :param fn: each line: e.g., [the 0.418 0.24968 -0.41242 ...]
    """
    vocab_word = Vocab()

    if fn is None:
        say('\nRandom Initialized Word Embeddings')
        return vocab_word, None

    say('\nLoad Initial Word Embedding...')
    emb = []
    with open(fn) as lines:
        for line in lines:
            line = line.strip().decode('utf-8').split()
            if len(line[1:]) != dim_emb or vocab_word.has_key(line[0]):
                continue
            vocab_word.add_word(line[0])
            emb.append(line[1:])

    emb = set_unk(vocab_word, emb)
    emb = np.asarray(emb, dtype=theano.config.floatX)
    vocab_word.add_word(UNK)

    assert emb.shape[0] == vocab_word.size(), 'emb: %d  vocab: %d' % (emb.shape[0], vocab_word.size())
    return vocab_word, emb


def set_unk(vocab_word, emb):
    if vocab_word.has_key(UNK):
        return emb
    unk = list(np.mean(np.asarray(emb, dtype=theano.config.floatX), 0))
    emb.append(unk)
    return emb


def move_data(src, dst):
    if not os.path.exists(dst):
        path = ''
        for dn in dst.split('/'):
            path += dn + '/'
            if os.path.exists(path):
                continue
            os.mkdir(path)
    if os.path.exists(os.path.join(dst, src)):
        os.remove(os.path.join(dst, src))
    shutil.move(src, dst)


def dump_data(data, fn):
    if not fn.endswith(".pkl.gz"):
        fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
    with gzip.open(fn, 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn is None:
        return None
    if not fn.endswith(".pkl.gz"):
        fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


def load_dir(dn):
    assert os.path.exists(dn)
    corpus = []
    file_names = os.listdir(dn)
    say('\nLoading the files: %s\n' % str(file_names))
    for fn in file_names:
        path = os.path.join(dn, fn)
        corpus.extend(load_data(path))
    return corpus


def load_results_dir(dn):
    assert os.path.exists(dn)
    corpus = []
    file_names = os.listdir(dn)
    say('\nLoading the files: %s\n' % str(file_names))
    for fn in file_names:
        path = os.path.join(dn, fn)
        corpus.append(load_data(path))
    return corpus
