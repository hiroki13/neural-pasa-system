import numpy as np


def show_case_dist(corpus):
    print '\nCASE DISTRIBUTION\n'

    """
    NAIST Ver 1.5
        DEV
            Ga: DEP: 7436, ZERO: 2665
            O : DEP: 5083, ZERO: 418
            Ni: DEP: 1612, ZERO: 137

        TEST
            Ga: DEP: 14074, ZERO: 4942
            O : DEP: 9485,  ZERO: 830
            Ni: DEP: 2517,  ZERO: 251
    """

    case_types = np.zeros((3, 5))
    n_prds = 0

    for doc in corpus:
        for sent in doc:
            for w in sent:
                if w.is_prd is False:
                    continue
                flag = False
                w_case_arg_ids = w.case_arg_ids
                w_case_types = w.case_types
                for i, (a, t) in enumerate(zip(w_case_arg_ids, w_case_types)):
                    if t > -1:
                        case_types[i][t] += 1
                        if 0 < t < 3:
                            flag = True
                if flag:
                    n_prds += 1

    for i, cases in enumerate(case_types):
        if i == 0:
            label = 'Ga'
        elif i == 1:
            label = 'O'
        else:
            label = 'Ni'
        print '\t%s\tBST: %d  DEP: %d  INTRA-ZERO: %d  INTER-ZEO: %d  EXOPHORA: %d' %\
              (label, cases[0], cases[1], cases[2], cases[3], cases[4])
    print '\n\tPredicates: %d' % n_prds


def corpus_statistics(corpus):
    if corpus is None:
        return

    print '\nCORPUS STATISTICS'

    """
    NAIST Ver. 1.5; DOC Train:1751, Dev:480, Test:696
    """
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
                    for case_i, arg in enumerate(word.case_arg_ids):
                        if arg > -1:
                            n_args += 1

    print '\tDocs: %d  Sents: %d  Words: %d' % (len(corpus), n_sents, n_words)
    print '\tPredicates: %d  Arguments %d' % (n_pds, n_args)
    print


def sample_statistics(samples, vocab_label):
    if samples is None:
        return

    print '\nSAMPLE STATISTICS'

    """
    The case distribution does not match with that of corpus_statistics(),
    because one word sometimes plays multiple case roles.
    Even in such cases, we assign one case role for a word.
    """

    n_samples = 0
    n_args = 0

    label_count = {}
    for key in vocab_label.w2i.keys():
        label_count[key] = 0
    n_labels = vocab_label.size()

    for sample in samples:
        sent = sample.label_ids
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


def inter_sample_statistics(samples):
    print '\nSAMPLE STATISTICS\n'

    """
    The case distribution does not match with that of corpus_statistics(),
    because one word sometimes plays multiple case roles.
    Even in such cases, we assign one case role for a word.
    """

    n_samples = len(samples)
    n_negs = 0

    for sample in samples:
        n_negs += len(sample.negative)

    print '\tSamples: Pos: %d  Neg: %d  Avg. Neg: %f\n' % (n_samples, n_negs, n_negs / float(n_samples))


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
