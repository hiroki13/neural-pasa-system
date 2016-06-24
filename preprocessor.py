import numpy as np
import theano

from vocab import Vocab


def get_sample_info(corpus, vocab_word, window=5):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents, 2D: n_prds, 3D: n_chars, 4D: [char_id, ctx_ids, prd_id, tag_id]
    """

    word_ids = []
    tag_ids = []
    prd_indices = []
    contexts = []
    vocab_tag = Vocab()

    sorted_corpus = []
    for doc in corpus:
        for sent in doc:
            sorted_corpus.append(sent)
    sorted_corpus.sort(key=lambda x: len(x))

    for sent in sorted_corpus:
        tmp_word_ids = [vocab_word.get_id(w.form) for w in sent]
        tmp_tag_ids, tmp_prd_indices, vocab_tag = get_tag_ids(sent, vocab_tag)

        if len(tmp_prd_indices) == 0:
            continue

        word_ids.append(tmp_word_ids)
        tag_ids.append(tmp_tag_ids)
        prd_indices.append(tmp_prd_indices)
        contexts.append(get_context(tmp_word_ids, tmp_prd_indices, window))

    assert len(word_ids) == len(tag_ids) == len(prd_indices) == len(contexts)
    return (word_ids, tag_ids, prd_indices, contexts), vocab_tag


def get_tag_ids(sent, vocab_tag):
    """
    :param sent: 1D: n_words, 2D: (word, pas_info, pas_id)
    :return: tag_ids: 1D: n_prds, 2D: n_chars; elem=tag_id
    :return: prd_indices: 1D: n_prds, 2D: n_chars of prd; elem=char index
    """

    tag_ids = []
    prd_indices = []

    for word in sent:
        if word.is_prd:  # check if the word is predicate or not
            prd_indices.append(word.index)
            p_tag_ids = []

            for arg in sent:
                case_label = None

                # case_arg_ids: [Ga_arg_id, O_arg_id, Ni_arg_id]
                for case_i, case_arg_id in enumerate(word.case_arg_ids):
                    if arg.pas_id > -1 and arg.pas_id == case_arg_id:
                        if case_i == 0:
                            case_label = 'Ga'
                        elif case_i == 1:
                            case_label = 'O'
                        elif case_i == 2:
                            case_label = 'Ni'
                        break

                if word.index == arg.index:
                    case_label = 'V'

                if case_label is None:
                    p_tag_ids.append(-1)
                else:
                    vocab_tag.add_word(case_label)
                    p_tag_ids.append(vocab_tag.get_id(case_label))

            assert len(p_tag_ids) == len(sent)
            tag_ids.append(p_tag_ids)

    return tag_ids, prd_indices, vocab_tag


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
    print '\nCORPUS STATISTICS'

    n_sents = 0
    n_words = 0
    n_pds = 0
    n_args = 0

    for doc in corpus:
        n_sents += len(doc)
        for sent in doc:
            n_words += len(sent)
            for word in sent:
                if word.is_prd:
                    n_pds += 1
                    for arg in word.case_arg_ids:
                        if arg > -1:
                            n_args += 1
    print '\tDocs: %d  Sents: %d  Words: %d' % (len(corpus), n_sents, n_words)
    print '\tPredicates: %d  Arguments %d' % (n_pds, n_args)


def theano_format(samples, batch_size=32):
    def shared(_sample):
        return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    theano_l = []
    theano_c = []
    batch_index = []
    sent_length = []

    sample_t = samples[1]
    sample_c = samples[-1]

    prev_n_words = len(sample_c[0][0])
    prev_index = 0
    index = 0

    for i in xrange(len(sample_c)):
        p_tags = sample_t[i]
        p_contexts = sample_c[i]  # 1D: n_prds, 2D: n_words, 3D: window; word_id

        """ Check the boundary of batches """
        n_words = len(p_contexts[0])
        if prev_n_words != n_words or index - prev_index > batch_size:
            batch_index.append((prev_index, index))
            sent_length.append(prev_n_words)
            prev_index = index

        for j in xrange(len(p_contexts)):
            labels = p_tags[j]
            contexts = p_contexts[j]

            for k in xrange(len(contexts)):
                label = labels[k]
                context = contexts[k]
                theano_l.append(label)
                theano_c.append(context)
                index += 1

    assert len(batch_index) == len(sent_length)
    return [shared(theano_c), shared(theano_l), shared(sent_length)], batch_index

