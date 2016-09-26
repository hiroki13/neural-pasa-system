import numpy as np
import theano
import theano.tensor as T


def main():
    print test_get_path_prob_memm()


def test_get_path_prob_memm():
    from ..nn.seq_labeling import get_path_prob_memm

    # 1D: n_words, 2D: batch, 3D: n_labels (j)
    h_in = np.asarray([[[0, 3], [1, 0]],
                       [[1, 2], [0, 1]]],
                      dtype='float32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[0, 0], [0, 1]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.ftensor3()
    y = T.imatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[h, y, W],
                        outputs=get_path_prob_memm(h, y, W),
                        on_unused_input='ignore')
    # 1D: n_words, 2D: batch; sum of the all the label scores
    return f(h_in, y_in, W_in)


def test_get_state_score():
    from ..nn.seq_labeling import get_state_score

    # 1D: n_words, 2D: batch, 3D: n_labels (j)
    h_in = np.asarray([[[0, 3], [1, 0]],
                       [[1, 2], [0, 1]]],
                      dtype='float32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[0, 0], [0, 1]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.ftensor3()
    y = T.imatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[h, y, W],
                        outputs=get_state_score(h, y, W),
                        on_unused_input='ignore')
    # 1D: n_words, 2D: batch; sum of the all the label scores
    return f(h_in, y_in, W_in)


def test_get_state_score_z():
    from ..nn.seq_labeling import get_state_score_z

    # 1D: n_words, 2D: batch, 3D: n_labels (j)
    h_in = np.asarray([[[0, 3], [1, 0]],
                       [[1, 2], [0, 1]]],
                      dtype='float32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[0, 0], [0, 1]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.ftensor3()
    y = T.imatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[h, y, W],
                        outputs=get_state_score_z(h, y, W),
                        on_unused_input='ignore')
    # 1D: n_words, 2D: batch; sum of the all the label scores
    return f(h_in, y_in, W_in)


def test_get_transition_scores():
    from ..nn.seq_labeling import get_transition_scores

    # 1D: n_words, 2D: batch
    y_in = np.asarray([[0, 0], [0, 1], [1, 0]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    y = T.imatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[y, W],
                        outputs=get_transition_scores(y, W),
                        on_unused_input='ignore')
    return f(y_in, W_in)


def test_get_path_prob_crf():
    from ..nn.seq_labeling import get_path_prob_crf

    # 1D: n_words, 2D: batch, 3D: n_labels (j)
    h_in = np.asarray([[[0, 0], [1, 0]],
                       [[0, 0], [0, 1]]],
                      dtype='float32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[0, 0], [0, 1]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 1], [1, 1]], dtype='float32')

    h = T.ftensor3()
    y = T.imatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[h, y, W],
                        outputs=get_path_prob_crf(h, y, W),
                        on_unused_input='ignore')
    return f(h_in, y_in, W_in)


def test_get_path_score():
    """
    :return: [26, 35, 39]
    """
    from ..nn.seq_labeling import get_path_score, get_emit_score, get_transition_score

    # 1D: n_words, 2D: batch, 3D: n_labels
    h_in = np.asarray([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]],
                       [[13, 14], [15, 16], [17, 18]]],
                      dtype='int32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j)
    W_in = np.asarray([[1, 2], [3, 4]], dtype='int32')

    h = T.itensor3()
    y = T.imatrix()
    W = T.imatrix()

    f = theano.function(inputs=[h, y, W],
                        outputs=get_path_score(h, y, W),
                        on_unused_input='ignore')
    e = theano.function(inputs=[h, y],
                        outputs=get_emit_score(h, y),
                        on_unused_input='ignore')
    t = theano.function(inputs=[y, W],
                        outputs=get_transition_score(y, W))
    return f(h_in, y_in, W_in), e(h_in, y_in), t(y_in, W_in)


def test_get_emit_score():
    """
    :return: [[2 (0, 0, 1), 3 (0, 1, 0), 5 (0, 2, 0)], [7, 10, 12], [13, 16, 17]]
    """
    from ..nn.seq_labeling import get_emit_score

    # 1D: n_words, 2D: batch, 3D: n_labels
    h_in = np.asarray([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]],
                       [[13, 14], [15, 16], [17, 18]]],
                      dtype='int32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype='int32')

    h = T.itensor3()
    y = T.imatrix()

    f = theano.function(inputs=[h, y],
                        outputs=get_emit_score(h, y),
                        on_unused_input='ignore')
    return f(h_in, y_in)


def test_get_transition_score():
    """
    :return: [[3, 2, 2], [1, 4, 3]]
    """
    from ..nn.seq_labeling import get_transition_score

    # 1D: n_words, 2D: batch; label id
    y_in = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype='int32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='int32')

    y = T.imatrix()
    W = T.imatrix()

    f = theano.function(inputs=[y, W],
                        outputs=get_transition_score(y, W))
    return f(y_in, W_in)


def test_viterbi_search():
    from ..nn.seq_labeling import viterbi_search

    # 1D: batch, 2D: n_labels (j)
    h_in = np.asarray([[[2, 1], [4, 3], [6, 5]], [[4, 1], [2, 3], [3, 5]]], dtype='float32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 1], [1, 1]], dtype='float32')

    h = T.ftensor3()
    W = T.fmatrix()

    f = theano.function(inputs=[h, W],
                        outputs=viterbi_search(h, W),
                        on_unused_input='ignore')

    return f(h_in, W_in)


def test_forward_viterbi():
    from ..nn.seq_labeling import forward_viterbi

    # 1D: batch, 2D: n_labels (j)
    h_in = np.asarray([[1, 2], [3, 4], [5, 6]], dtype='float32')
    # 1D: batch, 2D: n_labels (i)
    scores_prev_in = np.asarray([[1, 2], [3, 4], [5, 6]], dtype='float32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.fmatrix()
    scores_prev = T.fmatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[h, scores_prev, W],
                        outputs=forward_viterbi(h, scores_prev, W),
                        on_unused_input='ignore')

    return f(h_in, scores_prev_in, W_in)


def test_forward_alpha():
    """
        float32 is not precise compared with float64
        If you want more precise scores, use float64 and T.dmatrix()

        Returned matrix using float32:
        [[  6.04858732   8.0485878 ]
         [ 10.0485878   12.0485878 ]
         [ 14.0485878   16.0485878 ]]

        Returned matrix using float64:
        [[  6.0485878   8.0485878 ]
         [ 10.0485878   12.0485878 ]
         [ 14.0485878   16.0485878 ]]
    """
    from ..nn.seq_labeling import forward_alpha

    # 1D: batch, 2D: n_labels
    h_in = np.asarray([[1, 2], [3, 4], [5, 6]], dtype='float32')
    # 1D: batch, 2D: n_labels
    scores_prev_in = np.asarray([[1, 2], [3, 4], [5, 6]], dtype='float32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.fmatrix()
    scores_prev = T.fmatrix()
    W = T.fmatrix()

    f = theano.function(inputs=[h, scores_prev, W],
                        outputs=forward_alpha(h, scores_prev, W),
                        on_unused_input='ignore')

    return f(h_in, scores_prev_in, W_in)


def test_get_path_score_z():
    """
    :return: [  7.44018936  11.44019032  15.44019032]
    """
    from ..nn.seq_labeling import get_path_score_z

    # 1D: n_words, 2D: batch, 3D: n_labels
    h_in = np.asarray([[[2, 1], [4, 3], [6, 5]], [[1, 2], [3, 4], [5, 6]]], dtype='float32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    W_in = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.ftensor3()
    W = T.fmatrix()

    f = theano.function(inputs=[h, W],
                        outputs=get_path_score_z(h, W),
                        on_unused_input='ignore')

    return f(h_in, W_in)


if __name__ == '__main__':
    main()

