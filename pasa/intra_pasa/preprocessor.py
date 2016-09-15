import numpy as np
import theano

from ling.sample import Sample


def get_samples(corpus, vocab_word, vocab_label, window=5, test=False):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents; Sample
    """

    # 1D: n_docs * n_sents, 2D: n_words; elem=Word
    corpus = [sent for doc in corpus for sent in doc]

    ############################################
    # Arrange the training corpus with n_words #
    ############################################
    if test is False:
        np.random.shuffle(corpus)
        corpus.sort(key=lambda s: len(s))

    ###############
    # Set samples #
    ###############
    samples = []
    for sent in corpus:
        sample = Sample(sent=sent, window=window)
        sample.set_params(vocab_word, vocab_label)
        samples.append(sample)

    return samples


def get_shared_samples(samples, batch_size=32, mp=False):
    """
    :param samples: 1D: n_sents; Sample
    """

    def shared(_sample):
        return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    #######################################
    # Remove the samples that has no prds #
    #######################################
    samples = [sample for sample in samples if sample.n_prds > 0]

    ##########################################
    # Transform samples into shared_variable #
    ##########################################
    theano_x = []
    theano_y = []
    sent_length = []
    num_prds = []
    batch_index = []

    ##################
    # Initial values #
    ##################
    prev_n_prds = samples[0].n_prds
    prev_n_words = samples[0].n_words
    prev_index = 0
    index = 0

    for sample in samples:
        n_prds = sample.n_prds
        n_words = sample.n_words

        #################################
        # Check the boundary of batches #
        #################################
        if prev_n_words != n_words or index - prev_index > batch_size or (n_prds != prev_n_prds and mp):
            batch_index.append((prev_index, index))
            num_prds.append(prev_n_prds)
            sent_length.append(prev_n_words)
            prev_index = index
            prev_n_prds = n_prds
            prev_n_words = n_words

        ######################################
        # Add each sequence into the dataset #
        ######################################
        theano_x.extend(sample.x)
        theano_y.extend(sample.y)
        index += len(sample.x)

    if index > prev_index:
        batch_index.append((prev_index, index))
        num_prds.append(prev_n_prds)
        sent_length.append(prev_n_words)

    assert len(batch_index) == len(sent_length) == len(num_prds)
    return [shared(theano_x), shared(theano_y), shared(sent_length), shared(num_prds)], batch_index
