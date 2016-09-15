import numpy as np
import theano

from ling.vocab import Vocab

UNK = u'<UNK>'

Ga = 1
O = 2
Ni = 3


def get_ids(sent, vocab):
    ids = []
    for w in sent:
        for c in w.chars:
            if c not in vocab.w2i:
                c_id = vocab.get_id(UNK)
            else:
                c_id = vocab.get_id(c)
            ids.append(c_id)
    return ids


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


def get_samples(corpus, vocab_word, vocab_label=Vocab(), window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    word_ids = []
    labels = []
    prd_indices = []
    contexts = []

    sorted_corpus = []
    for doc in corpus:
        for sent in doc:
            sorted_corpus.append(sent)
    sorted_corpus.sort(key=lambda x: len(x))

    for sent in sorted_corpus:
        """ Get char ids """
        tmp_word_ids = get_ids(sent, vocab_word)

        """ Get labels """
        tmp_labels, tmp_prd_indices = get_labels(sent, vocab_label)

        if len(tmp_prd_indices) == 0:
            continue

        word_ids.append(tmp_word_ids)
        labels.append(tmp_labels)
        prd_indices.append(tmp_prd_indices)
        contexts.append(get_context(tmp_word_ids, tmp_prd_indices, window))

    assert len(word_ids) == len(labels) == len(prd_indices) == len(contexts)
    return word_ids, labels, prd_indices, contexts


def get_labels(sent, vocab_label):
    """
    :param sent: 1D: n_words, 2D: (word, pas_info, pas_id)
    :return: tag_ids: 1D: n_prds, 2D: n_chars; elem=tag_id
    :return: prd_indices: 1D: n_prds, 2D: n_chars of prd; elem=char index
    """

    labels = []
    prd_indices = []

    char_index = 0
    for word in sent:
        if word.is_prd:  # check if the word is predicate or not
            p_labels = []

            for arg in sent:
                case_label = None

                # case_arg_ids: [Ga_arg_id, O_arg_id, Ni_arg_id]
                for case_i, case_arg_id in enumerate(word.case_arg_ids):
                    if arg.id > -1 and arg.id == case_arg_id:
                        if case_i+1 == Ga:
                            case_label = 'Ga'
                        elif case_i+1 == O:
                            case_label = 'O'
                        elif case_i+1 == Ni:
                            case_label = 'Ni'
                        break

                if word.index == arg.index:
                    case_label = 'V'

                if case_label is None:
                    case_label = 'NA'

                p_labels.extend(get_one_label(arg, case_label, vocab_label))

            for label in p_labels:
                if 0 < label < 7:
                    break
            else:
                continue

            labels.append(p_labels)
            prd_indices.append(char_index)
        char_index += len(word.chars)

    return labels, prd_indices


def get_one_label(word, case_label, vocab_label):
    labels = []
    if case_label == 'NA':
        return [vocab_label.get_id(case_label) for c in word.chars]

    for i, c in enumerate(word.chars):
        if i == 0:
            labels.append(vocab_label.get_id('B-' + case_label))
        else:
            labels.append(vocab_label.get_id('I-' + case_label))

    return labels


def get_context(sent_w_ids, prd_indices, window=5):
    """
    :param sent_w_ids: 1D: n_words; elem=word id
    :param prd_indices: 1D: n_prds; prd index
    :return: context: 1D: n_words, 2D: window+1; elem=word id
    """

    context = []
    slide = window/2
    sent_len = len(sent_w_ids)
    pad = [0 for i in xrange(slide)]
    sent_w_ids = pad + sent_w_ids + pad

    for prd_index in prd_indices:
        prd_ctx = sent_w_ids[prd_index: prd_index + window]
        p_context = []

        for i in xrange(sent_len):
            c = sent_w_ids[i: i + window] + prd_ctx
            c.append(get_mark(prd_index, i, window))
            p_context.append(c)
        context.append(p_context)

    assert len(context) == len(prd_indices)
    return context


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

    sample_x = samples[-1]
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

    assert len(batch_index) == len(sent_length)
    return [shared(theano_x), shared(theano_y), shared(sent_length)], batch_index

