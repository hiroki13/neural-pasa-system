import numpy as np
import theano

from vocab import Vocab

UNK = u'<UNK>'

Ga = 1
O = 2
Ni = 3


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
        vocab_label.add_word('NA')
        vocab_label.add_word('Ga')
        vocab_label.add_word('O')
        vocab_label.add_word('Ni')
        vocab_label.add_word('V')

    sorted_corpus = []
    for doc in corpus:
        for sent in doc:
            sorted_corpus.append(sent)
    sorted_corpus.sort(key=lambda x: len(x))

    for sent in sorted_corpus:
        tmp_word_ids = []
        for w in sent:
            """ Get word ids """
            if w.form not in vocab_word.w2i:
                w_id = vocab_word.get_id(UNK)
            else:
                w_id = vocab_word.get_id(w.form)

            tmp_word_ids.append(w_id)

        """ Get labels """
        tmp_labels, tmp_prd_indices = get_label(sent, vocab_label)

        if len(tmp_prd_indices) == 0:
            continue

        word_ids.append(tmp_word_ids)
        labels.append(tmp_labels)
        prd_indices.append(tmp_prd_indices)
        contexts.append(get_context(tmp_word_ids, tmp_prd_indices, window))

    assert len(word_ids) == len(labels) == len(prd_indices) == len(contexts)
    return (word_ids, labels, prd_indices, contexts), vocab_label


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
            p_labels = []

            for arg in sent:
                case_label = None

                # case_arg_ids: [Ga_arg_id, O_arg_id, Ni_arg_id]
                for case_i, case_arg_id in enumerate(word.case_arg_ids):
                    if arg.pas_id > -1 and arg.pas_id == case_arg_id:
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


def get_context(sent, prd_indices, window=5):
    """
    :param sent: 1D: n_words; elem=word id
    :param prd_indices: 1D: n_prds; prd index
    :return: context: 1D: n_words, 2D: window+1; elem=word id
    """

    context = []
    sent_len = len(sent)
    slide = window/2
    pad = [0 for i in xrange(slide)]
    sent = pad + sent + pad

    for prd_index in prd_indices:
        prd_ctx = sent[prd_index: prd_index+window]
        context.append([[sent[i]] + prd_ctx for i in xrange(sent_len)])

    assert len(context) == len(prd_indices)
    return context


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


def sample_statistics(samples):
    print 'SAMPLE STATISTICS'

    n_samples = 0
    n_args = 0
    case_dict = {'Ga': 0, 'O': 0, 'Ni': 0}

    for sent in samples:
        for prd_labels in sent:
            flag = False
            for label in prd_labels:
                if 0 < label < 4:
                    n_args += 1
                    if label == 1:
                        case_dict['Ga'] += 1
                    elif label == 2:
                        case_dict['O'] += 1
                    else:
                        case_dict['Ni'] += 1

                    flag = True
            if flag:
                n_samples += 1

    print '\tSamples: %d' % n_samples
    for case, count in case_dict.items():
        print '\t%s: %d' % (case, count),
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

