import numpy as np
import theano

from ling.sample import Sample


def get_samples(corpus, vocab_word, vocab_label, window=5, test=False):
    """
    :param corpus: 1D: n_docs, 2D: n_sents, 3D: n_words; elem=Word
    :return: samples: 1D: n_sents; Sample
    """

    corpus = [sent for doc in corpus for sent in doc]

    if test is False:
        corpus.sort(key=lambda s: len(s))

    samples = []
    for sent in corpus:
        sample = Sample(sent=sent, window=window)
        sample.set_params(vocab_word, vocab_label)
        samples.append(sample)

    return samples


def get_shared_samples(samples, batch_size=32):
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
    batch_index = []
    sent_length = []
    prev_n_words = samples[0].n_words
    prev_index = 0
    index = 0

    for sample in samples:
        n_words = sample.n_words

        #################################
        # Check the boundary of batches #
        #################################
        if prev_n_words != n_words or index - prev_index > batch_size:
            batch_index.append((prev_index, index))
            sent_length.append(prev_n_words)
            prev_index = index
            prev_n_words = n_words

        ######################################
        # Add each sequence into the dataset #
        ######################################
        theano_x.extend(sample.x)
        theano_y.extend(sample.y)
        index += len(sample.x)

    if index > prev_index:
        batch_index.append((prev_index, index))
        sent_length.append(prev_n_words)

    assert len(batch_index) == len(sent_length)
    return [shared(theano_x), shared(theano_y), shared(sent_length)], batch_index
