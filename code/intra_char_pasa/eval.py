import numpy as np


class Eval(object):

    def __init__(self):
        self.corrects = np.zeros(3, dtype='float32')
        self.results_sys = np.zeros(3, dtype='float32')
        self.results_gold = np.zeros(3, dtype='float32')

        self.all_corrects = 0.
        self.all_results_sys = 0.
        self.all_results_gold = 0.

        self.precision = np.zeros(3, dtype='float32')
        self.recall = np.zeros(3, dtype='float32')
        self.f1 = np.zeros(3, dtype='float32')

        self.all_precision = 0.
        self.all_recall = 0.
        self.all_f1 = 0.

        self.nll = 0.

    def update_results(self, batch_y_hat, batch_y):
        assert len(batch_y_hat) == len(batch_y)
        assert len(batch_y_hat[0]) == len(batch_y[0])

        for i in xrange(len(batch_y_hat)):
            y_spans = get_spans(batch_y[i])
            y_hat_spans = get_spans(batch_y_hat[i])

            for s1 in y_spans:
                span1 = s1[0]
                label1 = s1[1]

                for s2 in y_hat_spans:
                    span2 = s2[0]
                    label2 = s2[1]
                    if span1 == span2:
                        if 1 <= label1 <= 2 and 1 <= label2 <= 2:
                            self.corrects[0] += 1
                        elif 3 <= label1 <= 4 and 3 <= label2 <= 4:
                            self.corrects[1] += 1
                        elif 5 <= label1 <= 6 and 5 <= label2 <= 6:
                            self.corrects[2] += 1

                    if 1 <= label2 <= 2:
                        self.results_sys[0] += 1
                    elif 3 <= label2 <= 4:
                        self.results_sys[1] += 1
                    elif 5 <= label2 <= 6:
                        self.results_sys[2] += 1

                if 1 <= label1 <= 2:
                    self.results_gold[0] += 1
                elif 3 <= label1 <= 4:
                    self.results_gold[1] += 1
                elif 5 <= label1 <= 6:
                    self.results_gold[2] += 1

    def set_metrics(self):
        for c in xrange(3):
            self.precision[c] = self.corrects[c] / self.results_sys[c]
            self.recall[c] = self.corrects[c] / self.results_gold[c]
            self.f1[c] = 2 * self.precision[c] * self.recall[c] / (self.precision[c] + self.recall[c])

        self.all_corrects = np.sum(self.corrects)
        self.all_results_sys = np.sum(self.results_sys)
        self.all_results_gold = np.sum(self.results_gold)

        self.all_precision = self.all_corrects / self.all_results_sys
        self.all_recall = self.all_corrects / self.all_results_gold
        self.all_f1 = 2 * self.all_precision * self.all_recall / (self.all_precision + self.all_recall)

    def show_results(self):
        self.set_metrics()
        print '\n\tEVALUATION'

        print '\t\tNLL: %f' % self.nll

        for c in xrange(3):
            print '\n\tCASE: %d' % c
            print '\t\tCRR: %d  TTL P: %d  TTL R: %d' % (self.corrects[c], self.results_sys[c], self.results_gold[c])
            print '\t\tF: %f  P: %f  R: %f' % (self.f1[c], self.precision[c], self.recall[c])

        print '\n\tTOTAL'
        print '\t\tCRR: %d  TTL P: %d  TTL R: %d' % (self.all_corrects, self.all_results_sys, self.all_results_gold)
        print '\t\tF: %f  P: %f  R: %f' % (self.all_f1, self.all_precision, self.all_recall)

    def check_spans(self, samples, vocab_word, vocab_label):
        # samples: (word_ids, tag_ids, prd_indices, contexts)
        # word_ids: 1D: n_sents, 2D: n_words
        # tag_ids: 1D: n_sents, 2D: n_prds, 3D: n_words
        word_ids = samples[0]
        tag_ids = samples[1]

        for sent1, sent2 in zip(word_ids, tag_ids):
            for prds in sent2:
                for w, t in zip(sent1, prds):
                    print '%s/%s' % (vocab_word.get_word(w), vocab_label.get_word(t)),
                print
            print

    def check_num_spans(self, corpus, samples, vocab_label):
        # corpus: 1D: n_sents, 2D: n_chars, 3D: (char, pas_info, pas_id)
        # samples: (word_ids, tag_ids, prd_indices, contexts)
        # tag_ids: 1D: n_sents, 2D: n_prds, 3D: n_words
        tag_ids = samples[1]

        count = 0
        for sent in tag_ids:
            for prds in sent:
                for t in prds:
                    if t == vocab_label.get_id('B-V'):
                        count += 1
        print 'Span V: %d' % count

        count = 0
        for sents in corpus:
            for sent in sents:
                for w in sent:
                    if w.is_prd:
                        count += 1
        print 'Corpus V: %d' % count


def get_spans(y):
    spans = []

    for i, label in enumerate(y):
        # Ga, O, Ni: 1-6
        if label < 1 or 6 < label:
            continue

        if len(spans) == 0:
            spans.append(((i, i+1), label))
        else:
            prev = spans[-1]
            prev_span = prev[0]
            prev_label = prev[1]

            if prev_span[1] == i and (label == prev_label or (label == prev_label + 1 and label % 2 == 0)):
                spans.pop()
                spans.append(((prev_span[0], i+1), label))
            else:
                spans.append(((i, i+1), label))
    return spans

