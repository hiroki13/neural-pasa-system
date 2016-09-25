import theano
import theano.tensor as T

from utils import logsumexp, logsumexp_3d, sample_weights, relu


def get_path_prob_pointwise(h, y):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label probability
    :param y: 1D: n_words, 2D: batch
    :return: 1D: batch; log probability of the correct sequence
    """
    # 1D: n_words * batch, 2D: n_labels
    h_reshaped = h.reshape((h.shape[0] * h.shape[1], -1))
    # 1D: n_words * batch
    y = y.flatten()
    p_y_gold = T.log(h_reshaped[T.arange(y.shape[0]), y])
    # 1D: batch, 2D: n_words
    p_y_gold = p_y_gold.reshape((h.shape[0], h.shape[1]))
    return T.sum(p_y_gold, axis=0)


def greedy_search(h):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label probability
    :return: 1D: n_words, 2D: batch; label id
    """
    return T.argmax(h, axis=2)


def get_path_prob(h, y, trans_matrix):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :param trans_matrix: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: batch; log probability of a path
    """
    return get_path_score(h, y, trans_matrix) - get_all_path_score(h, trans_matrix)


def get_path_score(h, y, trans_matrix):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :param trans_matrix: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: batch; path score
    """
    emit_scores = get_emit_score(h, y)
    transition_scores = get_transition_score(y, trans_matrix)
    return T.sum(emit_scores, axis=0) + T.sum(transition_scores, axis=0)


def get_emit_score(h, y):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :return: 1D: n_words, 2D: batch; specified label score
    """
    # 1D: n_words * batch, 2D: n_labels
    h = h.reshape((h.shape[0] * h.shape[1], -1))
    return h[T.arange(h.shape[0]), y.ravel()].reshape(y.shape)


def get_transition_score(y, trans_matrix):
    """
    :param y: 1D: n_words, 2D: batch; label id
    :param trans_matrix: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: n_words-1, 2D: batch; transition score
    """
    return trans_matrix[y[T.arange(y.shape[0]-1)], y[T.arange(y.shape[0]-1)+1]]


def viterbi_search(h, trans_matrix):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param trans_matrix: : 1D: n_labels, 2D: n_labels; label score; transition score between two labels
    :return: 1D: n_words, 2D: batch
    """
    alpha_init = h[0]
    [alpha, nodes], _ = theano.scan(fn=forward_viterbi,
                                    sequences=[h[1:]],
                                    outputs_info=[alpha_init, None],
                                    non_sequences=trans_matrix)

    # 1D: batch
    node_last = T.argmax(alpha[-1], axis=1)
    seq, _ = theano.scan(fn=lambda beta_t, node_t, batch_size: beta_t[T.arange(batch_size), node_t],
                         sequences=T.cast(nodes[::-1], 'int32'),
                         outputs_info=T.cast(node_last, 'int32'),
                         non_sequences=[node_last.shape[0]])

    return T.concatenate([seq[::-1], node_last.dimshuffle('x', 0)], axis=0)


def forward_viterbi(h_t, scores_prev, trans_matrix):
    """
    :param h_t: 1D: batch, 2D: n_labels (j); label score
    :param scores_prev: 1D: batch, 2D: n_labels (i); score history sum
    :param trans_matrix: 1D: n_y (i), 2D, n_labels (j); transition score from i to j
    :return: 1D: batch, 2D: n_labels (j)
    :return: 1D: batch, 2D: n_labels (j)
    """
    h_t = h_t.dimshuffle(0, 'x', 1)
    scores_prev = scores_prev.dimshuffle(0, 1, 'x')
    # 1D: batch, 2D: n_labels (i), 3D: n_labels (j)
    scores = scores_prev + h_t + trans_matrix
    return T.max_and_argmax(scores, axis=1)


def get_all_path_score(h, trans_matrix):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param trans_matrix: : 1D: n_labels, 2D: n_labels; label score; transition score between two labels
    :return: 1D: batch
    """
    alpha_init = h[0]
    alpha, _ = theano.scan(fn=forward_alpha,
                           sequences=[h[1:]],
                           outputs_info=alpha_init,
                           non_sequences=trans_matrix)
    return logsumexp(alpha[-1], axis=1).ravel()


def forward_alpha(h_t, scores_prev, trans_matrix):
    """
    :param h_t: 1D: batch, 2D: n_labels (j); label score
    :param scores_prev: 1D: batch, 2D: n_labels (i); score history sum
    :param trans_matrix: 1D: n_y (i), 2D, n_labels (j); transition score from i to j
    :return: 1D: batch, 2D: n_labels (j)
    """
    h_t = h_t.dimshuffle(0, 'x', 1)
    scores_prev = scores_prev.dimshuffle(0, 1, 'x')
    return logsumexp_3d(scores_prev + h_t + trans_matrix)


class Layer(object):

    def __init__(self, n_i, n_labels):
        self.W = theano.shared(sample_weights(n_i, n_labels))
        self.params = [self.W]

    def forward(self, x, h):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels
        """
        # 1D: n_words, 2D: batch, 3D: n_labels
        h = relu(T.dot(T.concatenate([x, h], 2), self.W))
        # 1D: n_words * batch, 2D: n_labels
        h_reshaped = h.reshape((h.shape[0] * h.shape[1], h.shape[2]))
        return T.nnet.softmax(h_reshaped).reshape((h.shape[0], h.shape[1], -1))

    def get_y_prob(self, h, y):
        """
        :param h: 1D: n_words * batch, 2D: n_labels
        :param y: 1D: n_words, 2D: batch
        :return: 1D: batch; log probability of the correct sequence
        """
        return get_path_prob_pointwise(h, y)

    def decode(self, h):
        """
        :param h: 1D: n_words * batch, 2D: n_labels
        :return: 1D: batch, 2D: n_words; the highest scoring sequence (label id)
        """
        return greedy_search(h).dimshuffle(1, 0)


class CRFLayer(object):

    def __init__(self, n_i, n_labels):
        self.W = theano.shared(sample_weights(n_i, n_labels))
        self.W_trans = theano.shared(sample_weights(n_labels, n_labels))
        self.params = [self.W, self.W_trans]

    def forward(self, x, h):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels
        """
        return relu(T.dot(T.concatenate([x, h], 2), self.W))

    def get_y_prob(self, h, y):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels
        :param y: 1D: n_words, 2D: batch
        :return: 1D: batch; log probability of the correct sequence
        """
        return get_path_prob(h, y, self.W_trans)

    def decode(self, h):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels
        :return: 1D: batch, 2D: n_words; the highest scoring sequence (label id)
        """
        return viterbi_search(h, self.W_trans).dimshuffle(1, 0)


class CRFLayer2(object):

    def __init__(self, n_i, n_labels):
        self.W = theano.shared(sample_weights(n_i, n_labels))
        self.W_trans = theano.shared(sample_weights(n_labels, n_labels))
        self.BOS = theano.shared(sample_weights(n_labels))
        self.params = [self.W, self.W_trans, self.BOS]

        self.decode = self.vitabi

    def y_prob(self, h, y):
        """
        :param h: 1D: n_words, 2D: Batch, 3D: n_y
        :param y: 1D: n_words, 2D: Batch
        :return: gradient of cross entropy: 1D: Batch
        """

        ##############################
        # From BOS to the first word #
        ##############################
        y_score0 = self.BOS[y[0]] + h[0, T.arange(h.shape[1]), y[0]]  # 1D: Batch
        z_scores0 = self.BOS + h[0]  # 1D: Batch, 2D: n_y

        ###################################
        # From the second word to the end #
        ###################################
        [_, y_scores, z_scores], _ = theano.scan(fn=self.forward_step,
                                                 sequences=[h[1:], y[1:]],
                                                 outputs_info=[y[0], y_score0, z_scores0],
                                                 non_sequences=self.W_trans)

        #################
        # Normalization #
        #################
        y_score = y_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return y_score - z_score

    def forward_step(self, h_t, y_t, y_prev, y_score_prev, z_scores_prev, W_trans):
        """
        :param h_t: 1D: Batch, 2D: n_y
        :param y_t: 1D: Batch
        :param y_prev: 1D: Batch
        :param y_score_prev: 1D: Batch
        :param z_scores_prev: 1D: Batch, 2D: n_y
        :param W_trans: 1D: n_y, 2D, n_y
        """
        y_score_t = y_score_prev + W_trans[y_t, y_prev] + h_t[T.arange(h_t.shape[0]), y_t]  # 1D: Batch
        z_sum = z_scores_prev.dimshuffle(0, 'x', 1) + W_trans  # 1D: Batch, 2D: n_y, 3D: n_y
        z_scores_t = logsumexp(z_sum, axis=2).reshape(h_t.shape) + h_t  # 1D: Batch, 2D: n_y
        return y_t, y_score_t, z_scores_t

    def vitabi(self, h):
        scores0 = self.BOS + h[0]
        [max_scores, max_nodes], _ = theano.scan(fn=self.forward,
                                                 sequences=[h[1:]],
                                                 outputs_info=[scores0, None],
                                                 non_sequences=self.W_trans)

        max_last_node = T.cast(T.argmax(max_scores[-1], axis=1), dtype='int32')

        nodes, _ = theano.scan(fn=self.backward,
                               sequences=max_nodes[::-1],
                               outputs_info=max_last_node)

        return T.concatenate([nodes[::-1].dimshuffle(1, 0), max_last_node.dimshuffle((0, 'x'))], 1)

    def forward(self, h_t, score_prev, W_trans):
        """
        :param h_t: 1D: Batch, 2D: n_y
        :param score_prev: 1D: Batch, 2D: n_y
        :param W_trans: 1D: n_y, 2D, n_y
        """
        score = score_prev.dimshuffle(0, 'x', 1) + W_trans.dimshuffle('x', 0, 1) + h_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_nodes_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, T.cast(max_nodes_t, dtype='int32')

    def backward(self, nodes_t, max_node_t):
        return nodes_t[T.arange(nodes_t.shape[0]), max_node_t]
