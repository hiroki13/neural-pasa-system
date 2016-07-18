import numpy as np
import theano
import random

from ling.vocab import Vocab

UNK = u'<UNK>'

Ga = 1
O = 2
Ni = 3


def set_init_vocab_labels(vocab_label):
    vocab_label.add_word('NA')
    vocab_label.add_word('Ga')
    vocab_label.add_word('O')
    vocab_label.add_word('Ni')
    vocab_label.add_word('V')
    return vocab_label


def get_ids(sent, vocab, model):
    ids = []
    if model == 'word':
        """ Get word ids """
        for w in sent:
            if w.form not in vocab.w2i:
                w_id = vocab.get_id(UNK)
            else:
                w_id = vocab.get_id(w.form)
            ids.append(w_id)
    else:
        """ Get char ids """
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


def get_sorted_corpus(corpus, model):
    sorted_corpus = []
    for doc in corpus:
        for sent in doc:
            if model == 'word':
                sorted_corpus.append((len(sent), sent))
            else:
                char_len = 0
                for w in sent:
                    char_len += len(w.chars)
                sorted_corpus.append((char_len, sent))
    sorted_corpus.sort(key=lambda x: x[0])
    return [sent[1] for sent in sorted_corpus]


def get_sample_info(corpus, vocab_word, vocab_label=Vocab(), window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    word_ids = []
    labels = []
    prd_indices = []
    contexts = []

    if vocab_label.size() == 0:
        vocab_label = set_init_vocab_labels(vocab_label)

    sorted_corpus = []
    for doc in corpus:
        for sent in doc:
            sorted_corpus.append(sent)
    sorted_corpus.sort(key=lambda x: len(x))

    for sent in sorted_corpus:
        """ Get word (char) ids """
        tmp_word_ids = get_ids(sent, vocab_word)

        """ Get labels """
        if model == 'word':
            tmp_labels, tmp_prd_indices = get_label(sent, vocab_label)
        else:
            tmp_labels, tmp_prd_indices = get_label_char(sent, vocab_label)

        if len(tmp_prd_indices) == 0:
            continue

        word_ids.append(tmp_word_ids)
        labels.append(tmp_labels)
        prd_indices.append(tmp_prd_indices)
        contexts.append(get_context(tmp_word_ids, tmp_prd_indices, window))

    assert len(word_ids) == len(labels) == len(prd_indices) == len(contexts)
    return (word_ids, labels, prd_indices, contexts), vocab_label


def get_inter_samples(corpus, vocab_word, model='word', vocab_label=Vocab(), window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    x = []
    y = []

    if vocab_label.size() == 0:
        vocab_label = set_init_vocab_labels(vocab_label, model)

    corpus_ids = get_corpus_ids(corpus, vocab_word)
    for i, doc in enumerate(corpus):
        for sent in doc:
            x_i, y_i = get_one_inter_sample(sent, doc, corpus_ids[i], window)

            if len(y_i) == 0:
                continue

            x.extend(x_i)
            y.extend(y_i)

    assert len(x) == len(y)
    return (x, y), vocab_label


"""
def get_label(sent, vocab_label):

    labels = []
    prd_indices = []

    for word in sent:
        if word.is_prd:  # check if the word is predicate or not
            p_labels = []

            for arg in sent:
                case_label = None

                # case_arg_ids: [Ga_arg_id, O_arg_id, Ni_arg_id]
                for case_i, case_arg_id in enumerate(word.case_arg_ids):
                    if word.chunk_index != arg.chunk_index and arg.id > -1 and arg.id == case_arg_id:
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

                p_labels.append(vocab_label.get_id(case_label))

            assert len(p_labels) == len(sent)
            for label in p_labels:
                if 0 < label < 4:
                    break
            else:
                continue

            labels.append(p_labels)
            prd_indices.append(word.index)

    return labels, prd_indices
"""

def get_label(sent, vocab_label):
    """
    :param sent: 1D: n_words, 2D: (word, pas_info, pas_id)
    :return: tag_ids: 1D: n_prds, 2D: n_chars; elem=tag_id
    :return: prd_indices: 1D: n_prds, 2D: n_chars of prd; elem=char index
    """

    labels = []
    prd_indices = []

    for word in sent:
        if word.is_prd:  # check if the word is predicate or not
            p_labels = [0 for i in xrange(len(sent))]
            p_labels[word.index] = 4

            is_arg = False
            for case_label, arg_index in enumerate(word.case_arg_index):
                if arg_index > -1:
                    p_labels[arg_index] = case_label + 1
                    is_arg = True

            if is_arg:
                labels.append(p_labels)
                prd_indices.append(word.index)

    return labels, prd_indices


def get_one_inter_sample(sent, doc, doc_ids, window=5, case=0):
    """
    :param sent: 1D: n_words; elem=Word()
    :param doc: 1D: n_sents, 2D: n_words; elem=Word()
    :param doc_ids: 1D: n_sents, 2D: n_words; elem=word id
    :return: x:
    :return: y:
    """

    x = []
    y = []

    for word in sent:
        if word.is_prd is False:  # check if the word is predicate or not
            continue

        for case_label, arg_indices in enumerate(word.inter_case_arg_index):
            if case_label != case:
                continue

            for arg_index in arg_indices:  # [(doc_index, index), ...]
                arg = doc[arg_index[0]][arg_index[1]]
                x.append(get_feature(word, arg, doc_ids, window))
                y.append(1)

                x_neg = get_neg_feature(word, arg, doc, doc_ids, 1, window)
                x.extend(x_neg)
                y.extend([0 for i in xrange(len(x_neg))])

    return x, y


def get_neg_feature(prd, arg, doc, doc_ids, n_negs=1, window=5):
    neg_feature = []
    doc_indices = range(prd.sent_index)
    for i in xrange(n_negs):
        random.shuffle(doc_indices)
        sent = doc[doc_indices[0]]
        sent_indices = range(len(sent))
        if len(sent_indices) < 2:
            continue
        random.shuffle(sent_indices)
        arg_neg = sent[sent_indices[0]] if sent_indices[0] != arg.index else sent[sent_indices[1]]
        neg_feature.append(get_feature(prd, arg_neg, doc_ids, window))
    return neg_feature


def get_feature(prd, arg, doc, window=5):
    slide = window/2
    pad = [0 for i in xrange(slide)]

    sent_p = doc[prd.sent_index]
    sent_p_ids = pad + sent_p + pad

    sent_a = doc[arg.sent_index]
    sent_a_ids = pad + sent_a + pad

    return sent_a_ids[arg.index: arg.index + window] + sent_p_ids[arg.index: arg.index + window]


def get_label_char(sent, vocab_label):
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

                p_labels.extend(get_char_label(arg, case_label, vocab_label))

            for label in p_labels:
                if 0 < label < 7:
                    break
            else:
                continue

            labels.append(p_labels)
            prd_indices.append(char_index)
        char_index += len(word.chars)

    return labels, prd_indices


def get_char_label(word, case_label, vocab_label):
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
#            c = [sent_w_ids[i + slide]] + prd_ctx
            c = sent_w_ids[i: i + window] + prd_ctx
            c.append(get_mark(prd_index, i, window))
#            c.append(distance(prd_index, i))
            p_context.append(c)
        context.append(p_context)

    assert len(context) == len(prd_indices)
    return context


def get_mark(prd_index, arg_index, window):
    slide = window / 2
    if prd_index - slide <= arg_index <= prd_index + slide:
        return 1
    return 2


def distance(prd_index, arg_index):
    diff = prd_index - arg_index
    if diff == 0 or diff == 1 or diff == 2 or diff == 3:
        return diff + 1
    elif 3 < diff < 11:
        return 5
    elif 11 <= diff:
        return 6
    elif diff == -1 or diff == -2 or diff == -3:
        return diff * -1 + 6
    elif -11 < diff < -3:
        return 10
    else:
        return 11


def corpus_statistics(corpus):
    print 'CORPUS STATISTICS'

    """
    NAIST Ver. 1.5; DOC Train:1751, Dev:480, Test:696
    """
    n_sents = 0
    n_words = 0
    n_pds = 0
    n_args = 0
    case_dict = {'Ga': 0, 'O': 0, 'Ni': 0, 'E_Ga': 0, 'E_O': 0, 'E_Ni': 0}

    for doc in corpus:
        n_sents += len(doc)
        for sent in doc:
            n_words += len(sent)
            for word in sent:
                if word.is_prd:
                    n_pds += 1
                    for case_i, arg in enumerate(word.case_arg_ids):
                        if arg > -1:
                            n_args += 1
                            if case_i == 0:
                                if arg < 1000:
                                    case_dict['Ga'] += 1
                                else:
                                    case_dict['E_Ga'] += 1
                            elif case_i == 1:
                                if arg < 1000:
                                    case_dict['O'] += 1
                                else:
                                    case_dict['E_O'] += 1
                            else:
                                if arg < 1000:
                                    case_dict['Ni'] += 1
                                else:
                                    case_dict['E_Ni'] += 1

    print '\tDocs: %d  Sents: %d  Words: %d' % (len(corpus), n_sents, n_words)
    print '\tPredicates: %d  Arguments %d' % (n_pds, n_args)
    for case, count in case_dict.items():
        print '\t%s: %d' % (case, count),
    print


def check_samples(samples, vocab_word, vocab_label):
    # samples: (word_ids, tag_ids, prd_indices, contexts)
    # word_ids: 1D: n_sents, 2D: n_words
    # tag_ids: 1D: n_sents, 2D: n_prds, 3D: n_words
    # prd_indices: 1D: n_sents, 2D: n_prds
    # contexts: 1D: n_sents, 2D: n_prds, 3D: n_words, 4D: window + 2
    labels = samples[1]
    contexts = samples[-1]
    assert len(labels) == len(contexts), '%d %d' % (len(labels), len(contexts))

    for sent_labels, sent_context in zip(labels, contexts):
        assert len(sent_labels) == len(sent_context), '%d %d' % (len(sent_labels), len(sent_context))

        for p_labels, p_context in zip(sent_labels, sent_context):
            assert len(p_labels) == len(p_context), '%d %d' % (len(p_labels), len(p_context))
            for label, context in zip(p_labels, p_context):
                for c in context:
                    print vocab_word.get_word(c),
                print vocab_label.get_word(label)
            print
    exit()


def sample_statistics(samples, vocab_label):
    print 'SAMPLE STATISTICS'

    n_samples = 0
    n_args = 0

    label_count = {}
    for key in vocab_label.w2i.keys():
        label_count[key] = 0
    n_labels = vocab_label.size()

    for sent in samples:
        for prd_labels in sent:
            flag = False
            for label in prd_labels:
                label_count[vocab_label.get_word(label)] += 1
                if 0 < label < n_labels-1:
                    n_args += 1
                    flag = True
            if flag:
                n_samples += 1

    print '\tSamples: %d' % n_samples
    print '\t',
    for case, count in label_count.items():
        print '%s: %d  ' % (case, count),
    print


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


def theano_format_inter(samples, batch_size=32):
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

