from ..utils.io_utils import say


class IOManager(object):

    def __init__(self, argv, vocab_word, vocab_label):
        self.argv = argv
        self.vocab_word = vocab_word
        self.vocab_label = vocab_label

    def output_results(self, fn, results, samples):
        assert len(results) == len(samples)

        vocab_label = self.vocab_label

        with open(fn, 'w') as fout:
            for sent_index in xrange(len(samples)):
                sample = samples[sent_index]
                result = results[sent_index]
                g_result = sample.label_ids

                #################
                # Raw sentences #
                #################
                text = 'SENT-%d ' % (sent_index + 1)
                for i, w in enumerate(sample.sent):
                    text += '%d:%s ' % (i, w.form)
                print >> fout, text.encode('utf-8')

                ################
                # PASA results #
                ################
                for i in xrange(len(result)):
                    r = result[i]
                    g_r = g_result[i]

                    prd_index = sample.prd_indices[i]
                    prd = sample.sent[prd_index]

                    ########
                    # Gold #
                    ########
                    text = 'GOLD-%d %d:%s\t' % (i+1, prd_index, prd.form)
                    for w_index, label in enumerate(g_r):
                        if 0 < label < 4:
                            word = sample.sent[w_index]
                            text += '%s:%d:%s ' % (vocab_label.get_word(label), w_index, word.form)
                    print >> fout, text.encode('utf-8')

                    ##########
                    # System #
                    ##########
                    text = 'PRED-%d %d:%s\t' % (i+1, prd_index, prd.form)
                    for w_index, label in enumerate(r):
                        if 0 < label < 4:
                            word = sample.sent[w_index]
                            text += '%s:%d:%s ' % (vocab_label.get_word(label), w_index, word.form)
                    print >> fout, text.encode('utf-8')

                print >> fout

