import numpy as np
import theano
import theano.tensor as T


def main():
    print test_get_emit_score()


def test_get_emit_score():
    """
    :return: [[2 (0, 0, 1), 3 (0, 1, 0), 5 (0, 2, 0)], [7, 10, 12], [13, 16, 17]]
    """
    from ..nn.crf import get_emit_score

    # 1D: n_words, 2D: batch, 3D: n_labels
    h_in = np.asarray([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]]], dtype='int32')
    # 1D: n_words, 2D: batch
    y_in = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype='int32')

    h = T.itensor3()
    y = T.imatrix()

    f = theano.function(inputs=[h, y],
                        outputs=get_emit_score(h, y),
                        on_unused_input='ignore')
    return f(h_in, y_in)


def test_get_transition_score():
    from ..nn.crf import get_transition_score

    M = np.asarray([[1, 2], [3, 4]], dtype='int32')
    y_in = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype='int32')

    trans_matrix = T.imatrix()
    y = T.imatrix()

    f = theano.function(inputs=[y, trans_matrix], outputs=get_transition_score(y, trans_matrix))
    return f(M, y_in)


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
    from ..nn.crf import forward_alpha

    # 1D: batch, 2D: n_labels
    h_in = np.asarray([[1, 2], [3, 4], [5, 6]], dtype='float32')
    # 1D: batch, 2D: n_labels
    scores_prev_in = np.asarray([[1, 2], [3, 4], [5, 6]], dtype='float32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    M = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.fmatrix()
    scores_prev = T.fmatrix()
    trans_matrix = T.fmatrix()

    f = theano.function(inputs=[h, scores_prev, trans_matrix],
                        outputs=forward_alpha(h, scores_prev, trans_matrix),
                        on_unused_input='ignore')

    return f(h_in, scores_prev_in, M)


def test_get_all_path_score():
    """
    :return: [  7.44018936  11.44019032  15.44019032]
    """
    from ..nn.crf import get_all_path_score

    # 1D: n_words, 2D: batch, 3D: n_labels
    h_in = np.asarray([[[2, 1], [4, 3], [6, 5]], [[1, 2], [3, 4], [5, 6]]], dtype='float32')
    # 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    M = np.asarray([[1, 2], [3, 4]], dtype='float32')

    h = T.ftensor3()
    trans_matrix = T.fmatrix()

    f = theano.function(inputs=[h, trans_matrix],
                        outputs=get_all_path_score(h, trans_matrix),
                        on_unused_input='ignore')

    return f(h_in, M)


if __name__ == '__main__':
    main()

