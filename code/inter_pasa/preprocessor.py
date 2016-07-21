import numpy as np
import theano
import random

from ling.vocab import UNK
from sample import Sample


def get_corpus_ids(corpus, vocab):
    ids = []
    for doc in corpus:
        doc_ids = []
        for sent in doc:
            sent_ids = []
            for w in sent:
                if w.form not in vocab.w2i:
                    w_id = vocab.get_id(UNK)
                else:
                    w_id = vocab.get_id(w.form)
                sent_ids.append(w_id)
            doc_ids.append(sent_ids)
        ids.append(doc_ids)
    return ids


def get_samples(corpus, vocab_word, test=False, n_cands=1, window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    samples = []
    corpus_ids = get_corpus_ids(corpus, vocab_word)
    for i, doc in enumerate(corpus):
        for sent in doc:
            sample = get_one_sample(sent, doc, corpus_ids[i], test, window, n_cands)
            if sample:
                samples.extend(sample)
    return samples, corpus_ids


def get_one_sample(sent, doc, doc_ids, test=False, window=5, n_negs=1, case=0):
    """
    :param sent: 1D: n_words; elem=Word()
    :param doc: 1D: n_sents, 2D: n_words; elem=Word()
    :param doc_ids: 1D: n_sents, 2D: n_words; elem=word id
    :return: samples: 1D: n_samples; elem=Sample()
    """

    samples = []

    for word in sent:
        if word.is_prd is False:  # check if the word is predicate or not
            continue

        indices = [(i,  j) for i in xrange(word.sent_index) for j in xrange(len(doc[i]))]
        for case_label, arg_indices in enumerate(word.inter_case_arg_index):
            if case_label != case:
                continue

            for arg_index in arg_indices:  # [(doc_index, index), ...]
                sample = Sample()
                arg = doc[arg_index[0]][arg_index[1]]
                sample.positive = get_one_phi(prd=word, arg=arg, doc_ids=doc_ids, window=window)
                if test:
                    sample.negative = get_all_phi(prd=word, doc=doc, doc_ids=doc_ids, window=window)
                else:
                    sample.negative = get_neg_phi(prd=word, arg=arg, doc=doc, doc_ids=doc_ids, indices=indices,
                                                  n_negs=n_negs, window=window)

                if sample.negative:
                    samples.append(sample)

    return samples


def get_one_phi(prd, arg, doc_ids, window=5):
    slide = window / 2
    pad = [0 for i in xrange(slide)]

    sent_p = doc_ids[prd.sent_index]
    sent_p_ids = pad + sent_p + pad
    sent_a = doc_ids[arg.sent_index]
    sent_a_ids = pad + sent_a + pad

    return sent_a_ids[arg.index: arg.index + window] + sent_p_ids[prd.index: prd.index + window]


def get_neg_phi(prd, arg, doc, doc_ids, indices, n_negs=1, window=5):
    neg_feature = []
    random.shuffle(indices)
    for i in xrange(n_negs):
        if i >= len(indices):
            break

        sent_index, word_index = indices[i]
        sent = doc[sent_index]
        arg_neg = sent[word_index]

        if arg_neg.index == arg.index and arg_neg.sent_index == arg.sent_index:
            continue

        neg_feature.append(get_one_phi(prd, arg_neg, doc_ids, window))

    return neg_feature


def get_all_phi(prd, doc, doc_ids, window=5):
    neg_feature = []
    for i in xrange(prd.sent_index):
        for arg in doc[i]:
            neg_feature.append(get_one_phi(prd, arg, doc_ids, window))
    return neg_feature


def theano_format(samples, batch_size=32): # TODO
    """
    :param samples: 1D: n_samples; elem=Sample()
    :return:
    """
    def shared(_sample):
        return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    x = []
    batch_index = []
    n_cands = []

    prev_index = 0
    index = 0

    samples.sort(key=lambda _sample: len(_sample.negative))
    prev_n_negs = len(samples[0].negative)

    for sample in samples:
        negs = sample.negative
        n_negs = len(negs)

        if prev_n_negs != n_negs or (index - prev_index) / prev_n_negs > batch_size:
            batch_index.append((prev_index, index))
            n_cands.append(prev_n_negs + 1)
            prev_index = index
            prev_n_negs = n_negs

        x.append(sample.positive)
        x.extend(sample.negative)
        index += 1 + n_negs

    if index > prev_index:
        batch_index.append((prev_index, index))
        n_cands.append(prev_n_negs + 1)

    return [shared(x), shared(n_cands)], batch_index

