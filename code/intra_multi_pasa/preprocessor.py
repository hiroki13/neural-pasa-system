import numpy as np
import theano

from ling.vocab import UNK, NA, GA, O, NI, PRD, GA_LABEL, O_LABEL


def get_ids(sent, vocab):
    ids = []
    for w in sent:
        if w.form not in vocab.w2i:
            w_id = vocab.get_id(UNK)
        else:
            w_id = vocab.get_id(w.form)
        ids.append(w_id)
    return ids


def get_samples(corpus, vocab_word, vocab_label, window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    word_ids = []
    xy = []
    y = []
    x = []

    sorted_corpus = [sent for doc in corpus for sent in doc]
    sorted_corpus.sort(key=lambda s: len(s))

    for sent in sorted_corpus:
        """ Get word ids """
        tmp_word_ids = get_ids(sent, vocab_word)

        """ Get labels """
        tmp_labels, tmp_prd_indices = get_labels(sent, vocab_label)

        if len(tmp_prd_indices) == 0:
            continue

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
    a_sent_w_ids = pad + sent_w_ids + pad

    p_window = 5
    p_slide = p_window / 2
    p_pad = [0 for i in xrange(p_slide)]
    p_sent_w_ids = p_pad + sent_w_ids + p_pad

    for prd_index in prd_indices:
        prd_ctx = p_sent_w_ids[prd_index: prd_index + p_window]
        p_phi = []

        for arg_index in xrange(sent_len):
            arg_ctx = a_sent_w_ids[arg_index: arg_index + window] + prd_ctx
            arg_ctx.append(get_mark(prd_index, arg_index))
            p_phi.append(arg_ctx)
        phi.append(p_phi)

    assert len(phi) == len(prd_indices)
    return phi


def get_labels(sent, vocab_label):
    """
    :param sent: 1D: n_words, 2D: (word, pas_info, pas_id)
    :return: labels: 1D: n_prds, 2D: n_chars; elem=label id
    :return: prd_indices: 1D: n_prds, 2D: n_chars of prd; elem=char index
    """

    labels = []
    prd_indices = []

    for word in sent:
        if word.is_prd:  # check if the word is a predicate or not
            p_labels = [vocab_label.get_id(NA) for i in xrange(len(sent))]
            p_labels[word.index] = vocab_label.get_id(PRD)

            is_arg = False
            for case_label, arg_index in enumerate(word.case_arg_index):
                if arg_index > -1:
                    if case_label == GA_LABEL:
                        p_labels[arg_index] = vocab_label.get_id(GA)
                    elif case_label == O_LABEL:
                        p_labels[arg_index] = vocab_label.get_id(O)
                    else:
                        p_labels[arg_index] = vocab_label.get_id(NI)
                    is_arg = True

            if is_arg:
                labels.append(p_labels)
                prd_indices.append(word.index)

    assert len(labels) == len(prd_indices)
    return labels, prd_indices


def get_mark(prd_index, arg_index, window=5):
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
    num_prds = []
    sent_length = []

    sample_x = samples[0]
    sample_y = samples[1]

    n_samples = len(sample_x)
    prev_n_prds = len(sample_x[0])
    prev_n_words = len(sample_x[0][0])
    prev_index = 0
    index = 0

    for i in xrange(n_samples):
        prd_x = sample_x[i]  # 1D: n_prds, 2D: n_words, 3D: window; word_id
        prd_y = sample_y[i]

        """ Check the boundary of batches """
        n_prds = len(prd_x)
        n_words = len(prd_x[0])
        if prev_n_prds != n_prds or prev_n_words != n_words or index - prev_index > batch_size:
            batch_index.append((prev_index, index))
            num_prds.append(prev_n_prds)
            sent_length.append(prev_n_words)
            prev_index = index
            prev_n_prds = n_prds
            prev_n_words = n_words

        for j in xrange(n_prds):
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
        num_prds.append(prev_n_prds)
        sent_length.append(prev_n_words)

    assert len(batch_index) == len(sent_length) == len(num_prds)
    return [shared(theano_x), shared(theano_y), shared(sent_length), shared(num_prds)], batch_index


def theano_format_online(samples):
    """
    :param samples: (sample_x, sample_y)
    :return: theano_x: 1D: n_sents, 2D: n_prds * n_words, 3D: window; word_id
    :return: theano_y: 1D: n_sents, 2D: n_prds * n_words
    :return: sent_length: 1D: n_sents; int
    """

    def numpize(_sample):
        return np.asarray(_sample, dtype='int32')

    theano_x = []
    theano_y = []
    sent_length = []

    # x: 1D: n_sents, 2D: n_prds, 3D: n_words, 4D: window; word_id
    # y: 1D: n_sents, 2D: n_prds, 3D: n_words; label
    sample_x = samples[0]
    sample_y = samples[1]

    for i in xrange(len(sample_x)):
        sent_x = []
        sent_y = []

        prd_x = sample_x[i]
        prd_y = sample_y[i]
        sent_length.append(len(prd_x[0]))

        for j in xrange(len(prd_x)):
            sent_x += prd_x[j]
            sent_y += prd_y[j]

        theano_x.append(numpize(sent_x))
        theano_y.append(numpize(sent_y))

    return theano_x, theano_y, numpize(sent_length)

