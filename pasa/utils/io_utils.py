import sys
import gzip
import cPickle

from ..ling.word import Word


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


class CorpusLoader(object):

    def __init__(self, min_unit, data_size):
        self.min_unit = min_unit
        self.data_size = data_size

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

                if line.startswith(BOD):  # Doc starts
                    doc_id = self.get_doc_id(elem)

                    if prev_doc_id and prev_doc_id != doc_id:
                        prev_doc_id = doc_id
                        corpus.append(doc)
                        doc = []
                    elif prev_doc_id is None:
                        prev_doc_id = doc_id

                elif line.startswith(BOC):
                    chunk_index, chunk_head = self.get_chunk_info(elem)

                elif line.startswith(EOS):
                    for w in sent:
                        w.set_cases(sent, doc)
                    doc.append(sent)
                    sent = []

                else:
                    word = self.get_word(w_index=len(sent),
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
    def get_chunk_info(elem):
        return int(elem[1]), int(elem[2][:-1])

    @staticmethod
    def get_doc_id(elem):
        return elem[1].split(':')[1].split('-')[0]

    @staticmethod
    def get_word(w_index, chunk_index, chunk_head, sent_index, elem):
        w = Word(w_index, elem)
        w.sent_index = sent_index
        w.chunk_index = chunk_index
        w.chunk_head = chunk_head
        return w


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn[-7:] != '.pkl.gz':
        fn += '.pkl.gz'

    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)
