import numpy as np


def eval_args(batch_y_hat, batch_y):
    assert len(batch_y_hat) == len(batch_y)
    assert len(batch_y_hat[0]) == len(batch_y[0])

    crr = np.zeros(3, dtype='float32')
    ttl_p = np.zeros(3, dtype='float32')
    ttl_r = np.zeros(3, dtype='float32')

    for i in xrange(len(batch_y_hat)):
        sent_y_hat = batch_y_hat[i]
        sent_y = batch_y[i]
        for j in xrange(len(sent_y_hat)):
            y_hat = sent_y_hat[j]
            y = sent_y[j]

            if 0 < y_hat == y < 4:
                crr[y_hat-1] += 1
            if 0 < y_hat < 4:
                ttl_p[y_hat-1] += 1
            if 0 < y < 4:
                ttl_r[y-1] += 1

    return crr, ttl_p, ttl_r


def eval_char_args(batch_y_hat, batch_y):
    assert len(batch_y_hat) == len(batch_y)
    assert len(batch_y_hat[0]) == len(batch_y[0])
    crr = np.zeros(3, dtype='float32')
    ttl_p = np.zeros(3, dtype='float32')
    ttl_r = np.zeros(3, dtype='float32')

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
                        crr[0] += 1
                    elif 3 <= label1 <= 4 and 3 <= label2 <= 4:
                        crr[1] += 1
                    elif 5 <= label1 <= 6 and 5 <= label2 <= 6:
                        crr[2] += 1

                if 1 <= label2 <= 2:
                    ttl_p[0] += 1
                elif 3 <= label2 <= 4:
                    ttl_p[1] += 1
                elif 5 <= label2 <= 6:
                    ttl_p[2] += 1

            if 1 <= label1 <= 2:
                ttl_r[0] += 1
            elif 3 <= label1 <= 4:
                ttl_r[1] += 1
            elif 5 <= label1 <= 6:
                ttl_r[2] += 1

    return crr, ttl_p, ttl_r


def get_spans(y):
    spans = []

    for i, label in enumerate(y):
        if label < 1 or label > 6:
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

