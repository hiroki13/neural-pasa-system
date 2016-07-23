import numpy as np
import theano

from ling.vocab import UNK


def get_ids(sent, vocab):
    ids = []
    for w in sent:
        if w.form not in vocab.w2i:
            w_id = vocab.get_id(UNK)
        else:
            w_id = vocab.get_id(w.form)
        ids.append(w_id)
    return ids


def get_samples(corpus, vocab_word, window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    word_ids = []
    xy = []

    corpus.sort(key=lambda s: len(s))

    for sent in corpus:
        """ Get word ids """
        tmp_word_ids = get_ids(sent, vocab_word)

        """ Get labels """
        tmp_labels = get_labels(sent)
        tmp_prd_indices = [word.index for word in sent if word.is_prd]

        if len(tmp_prd_indices) == 0 or len(sent) < 2:
            continue

        for i, p in enumerate(tmp_prd_indices):
            tmp_labels[i][p] = 1  # V

        word_ids.append(tmp_word_ids)
        xy.append((get_phi(tmp_word_ids, tmp_prd_indices, window), tmp_labels, (len(sent), len(tmp_labels))))

    xy.sort(key=lambda s: s[-1])
    x, y, _ = zip(*xy)
    assert len(word_ids) == len(x) == len(y)
    return (x, y), word_ids


def get_phi(sent_w_ids, prd_indices, window=5):
    """
    :param sent_w_ids: 1D: n_words; elem=word id
    :param prd_indices: 1D: n_prds; prd index
    :return: context: 1D: n_words, 2D: window+1; elem=word id
    """

    phi = []
    slide = window / 2
    sent_len = len(sent_w_ids)
    pad = [0 for i in xrange(slide)]
    sent_w_ids = pad + sent_w_ids + pad

    for prd_index in prd_indices:
        prd_ctx = sent_w_ids[prd_index: prd_index + window]
        p_phi = []

        for arg_index in xrange(sent_len):
            arg_ctx = sent_w_ids[arg_index: arg_index + window] + prd_ctx
            arg_ctx.append(get_mark(prd_index, arg_index, window))
            p_phi.append(arg_ctx)
        phi.append(p_phi)

    assert len(phi) == len(prd_indices)
    return phi


def get_labels(sent):
    """
    :param sent: 1D: n_words; elem=Word()
    :return: 1D: n_prds, 2D: n_words; elem=label id
    """
    return np.asarray([word.labels for word in sent], dtype='int32').T


def get_mark(prd_index, arg_index, window):
    slide = window / 2
    if prd_index - slide <= arg_index <= prd_index + slide:
        return 1
    return 2


def theano_format(samples, batch_size=32):
    def shared(_sample):
        return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    theano_x = []
    theano_y = []
    batch_index = []
    sent_length = []

    sample_x = samples[0]
    sample_y = samples[1]

    prev_n_words = len(sample_x[0][0])
    prev_index = 0
    index = 0

    for i in xrange(len(sample_x)):
        prd_x = sample_x[i]  # 1D: n_prds, 2D: n_words, 3D: window; word_id
        prd_y = sample_y[i]

        """ Check the boundary of batches """
        n_words = len(prd_x[0])
        if prev_n_words != n_words or index - prev_index > batch_size:
            batch_index.append((prev_index, index))
            sent_length.append(prev_n_words)
            prev_index = index
            prev_n_words = n_words

        for j in xrange(len(prd_x)):
            sent_x = prd_x[j]
            sent_y = prd_y[j]

            for k in xrange(len(sent_x)):
                x = sent_x[k]
                y = sent_y[k]
                theano_x.append(x)
                theano_y.append(y)
                index += 1

    if index > prev_index:
        batch_index.append((prev_index, index))
        sent_length.append(prev_n_words)

    assert len(batch_index) == len(sent_length)
    return [shared(theano_x), shared(theano_y), shared(sent_length)], batch_index
