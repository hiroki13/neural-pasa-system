import sys
import time
import numpy as np

import io_utils
from preprocessor import get_sample_info, theano_format, corpus_statistics
from model_builder import set_model, set_train_f


def train(argv):
    print 'SETTING UP A TRAINING SETTING'

    emb = None

    """ Load files """
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    corpus, vocab_word = io_utils.load_ntc(argv.train_data)
    corpus_statistics(corpus)

    """ Preprocessing """
    # samples: 1D: n_sents, 2D: [word_ids, tag_ids, prd_indices, contexts]
    # vocab_tags: {Ga:0, O:1, Ni:2, V:3}
    tr_dataset, vocab_tags = get_sample_info(corpus, vocab_word)

    # dataset: (labels, contexts, sent_length)
    tr_samples, batch_index = theano_format(tr_dataset)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    """ Set a model """
    print '\n\nBuilding a model...'
    model = set_model(argv=argv, emb=emb, vocab=vocab_word)
    train_f = set_train_f(model, tr_samples)

    ###############
    # TRAIN MODEL #
    ###############

    print '\nTraining start\n'
    indices = range(len(batch_index))

    for epoch in xrange(argv.epoch):
        print '\nEpoch: %d' % (epoch + 1)
        print '\tTRAIN\n\t',

        np.random.shuffle(indices)
        start = time.time()

        for index, b_index in enumerate(indices):
            if index != 0 and index % 100 == 0:
                print index,
                sys.stdout.flush()

            batch_range = batch_index[b_index]
            result = train_f(index=b_index, bos=batch_range[0], eos=batch_range[1])
            print result
            exit()

        end = time.time()
        print '\n\tTime: %f' % (end - start)
