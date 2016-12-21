import theano
import theano.tensor as T

from nn_utils import logsumexp, logsumexp_3d, sample_weights, relu


def get_path_prob_memm(h, y, W_trans):
    """
        p(y_n|x, y_n-1) = exp(f(x, y_n, y_n-1)) / sum_{y_n}(exp(f(x, y_n, y_n-1)))
        log p(y_n|x, y_n-1) = f(x, y_n, y_n-1) - log(sum_{y_n}(exp(f(x, y_n, y_n-1))))
        log p(y|x) = sum_{n}(log p(y_n|x, y_n-1))

    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: batch; log probability of a path
    """
    # 1D: n_words, 2D: batch; label score
    state_score = get_state_score(h, y, W_trans)
    # 1D: n_words, 2D: batch; specified label score
    state_score_z = get_state_score_z(h, y, W_trans).reshape(y.shape)
    return T.sum(state_score - state_score_z, axis=0)


def get_path_prob_crf(h, y, W_trans):
    """
        p(y|x) = exp(f(x, y)) / sum_{y}(exp(f(x, y)))
        log p(y|x) = f(x, y) - log(sum_{y}(exp(f(x, y))))

    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: batch; log probability of a path
    """
    return get_path_score(h, y, W_trans) - get_path_score_z(h, W_trans)


def get_state_score(h, y, W_trans):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: n_words, 2D: batch; label score
    """
    # 1D: n_words, 2D: batch; specified label score
    emit_score = get_emit_score(h, y)
    # 1D: n_words-1, 2D: batch; specified label score
    trans_score = get_transition_score(y, W_trans)

    zero = T.zeros(shape=(1, h.shape[1]), dtype=theano.config.floatX)
    # 1D: n_words, 2D: batch; label score
    trans_score = T.concatenate([zero, trans_score], axis=0)

    return emit_score + trans_score


def get_path_score(h, y, W_trans):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: batch; path score
    """
    emit_scores = get_emit_score(h, y)
    transition_scores = get_transition_score(y, W_trans)
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


def get_state_score_z(h, y, W_trans):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels (j); label score
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: n_words, 2D: batch; specified label score
    """
    # 1D: n_words-1, 2D: batch, 3D: n_labels (j); label score
    trans_scores = get_transition_scores(y, W_trans)
    # 1D: 1, 2D: batch, 3D: n_labels (j); 0
    zero = T.zeros(shape=(1, h.shape[1], h.shape[2]), dtype=theano.config.floatX)
    # 1D: n_words, 2D: batch, 3D: n_labels (j); label score
    trans_scores = T.concatenate([zero, trans_scores], axis=0)
    return logsumexp(h + trans_scores, axis=2)


def get_transition_score(y, W_trans):
    """
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: n_words-1, 2D: batch; transition score
    """
    return W_trans[y[T.arange(y.shape[0] - 1)], y[T.arange(y.shape[0] - 1) + 1]]


def get_transition_scores(y, W_trans):
    """
    :param y: 1D: n_words, 2D: batch; label id
    :param W_trans: 1D: n_labels (i), 2D: n_labels (j); transition score from i to j
    :return: 1D: n_words-1, 2D: batch, 3D: n_labels (j); label score
    """
    return W_trans[y[T.arange(y.shape[0] - 1)]]


def get_path_score_z(h, W_trans):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param W_trans: : 1D: n_labels, 2D: n_labels; label score; transition score between two labels
    :return: 1D: batch
    """
    alpha_init = h[0]
    alpha, _ = theano.scan(fn=forward_alpha,
                           sequences=[h[1:]],
                           outputs_info=alpha_init,
                           non_sequences=W_trans)
    return logsumexp(alpha[-1], axis=1).ravel()


def forward_alpha(h_t, scores_prev, W_trans):
    """
    :param h_t: 1D: batch, 2D: n_labels (j); label score
    :param scores_prev: 1D: batch, 2D: n_labels (i); score history sum
    :param W_trans: 1D: n_y (i), 2D, n_labels (j); transition score from i to j
    :return: 1D: batch, 2D: n_labels (j)
    """
    h_t = h_t.dimshuffle(0, 'x', 1)
    scores_prev = scores_prev.dimshuffle(0, 1, 'x')
    return logsumexp_3d(scores_prev + h_t + W_trans)


def viterbi_search(h, W_trans):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param W_trans: : 1D: n_labels, 2D: n_labels; label score; transition score between two labels
    :return: 1D: n_words, 2D: batch
    """
    alpha_init = h[0]
    [alpha, nodes], _ = theano.scan(fn=forward_viterbi,
                                    sequences=[h[1:]],
                                    outputs_info=[alpha_init, None],
                                    non_sequences=W_trans)

    # 1D: batch
    node_last = T.argmax(alpha[-1], axis=1)
    seq, _ = theano.scan(fn=lambda beta_t, node_t, batch_size: beta_t[T.arange(batch_size), node_t],
                         sequences=T.cast(nodes[::-1], 'int32'),
                         outputs_info=T.cast(node_last, 'int32'),
                         non_sequences=[node_last.shape[0]])

    return T.concatenate([seq[::-1], node_last.dimshuffle('x', 0)], axis=0)


def forward_viterbi(h_t, scores_prev, W_trans):
    """
    :param h_t: 1D: batch, 2D: n_labels (j); label score
    :param scores_prev: 1D: batch, 2D: n_labels (i); sum of the score history
    :param W_trans: 1D: n_y (i), 2D, n_labels (j); transition score from i to j
    :return: 1D: batch, 2D: n_labels (j)
    :return: 1D: batch, 2D: n_labels (j)
    """
    h_t = h_t.dimshuffle(0, 'x', 1)
    scores_prev = scores_prev.dimshuffle(0, 1, 'x')
    # 1D: batch, 2D: n_labels (i), 3D: n_labels (j)
    scores = scores_prev + h_t + W_trans
    return T.max_and_argmax(scores, axis=1)


def greedy_search(h, W_trans):
    """
    :param h: 1D: n_words, 2D: batch, 3D: n_labels; label score
    :param W_trans: : 1D: n_labels, 2D: n_labels; label score; transition score between two labels
    :return: 1D: n_words, 2D: batch
    """
    # 1D: batch
    state_init = T.argmax(h[0], axis=1)
    states, _ = theano.scan(fn=lambda h_t, state_prev, W: T.argmax(h_t + W[state_prev], axis=1),
                            sequences=[h[1:]],
                            outputs_info=state_init,
                            non_sequences=W_trans)
    return T.concatenate([state_init.dimshuffle('x', 0), states], axis=0)


class SoftmaxLayer(object):

    def __init__(self, n_i, n_labels):
        self.W = theano.shared(sample_weights(n_i, n_labels))
        self.params = [self.W]

    def forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels; log probability of a label
        """
        # 1D: n_words, 2D: batch, 3D: n_labels
        h = T.dot(x, self.W)
        # 1D: n_words * batch, 2D: n_labels
        h_reshaped = h.reshape((h.shape[0] * h.shape[1], h.shape[2]))
#        return T.log(T.nnet.softmax(h_reshaped).reshape((h.shape[0], h.shape[1], -1)))
        return T.nnet.softmax(h_reshaped).reshape((h.shape[0], h.shape[1], -1))

    @staticmethod
    def get_y_prob(h, y):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels
        :param y: 1D: n_words, 2D: batch
        :return: 1D: batch; log probability of the correct sequence
        """
        emit_scores = get_emit_score(h, y)
        return T.sum(emit_scores, axis=0)

    @staticmethod
    def decode(h):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels; log probability of a label
        :return: 1D: batch, 2D: n_words; the highest scoring sequence (label id)
        """
        return T.argmax(h, axis=2).dimshuffle(1, 0)


class MixedLayer(object):

    def __init__(self, n_i):
        self.W = theano.shared(sample_weights(n_i*2, n_i))
        self.W_a = theano.shared(sample_weights(n_i, n_i))
        self.W_p = theano.shared(sample_weights(n_i, n_i))
        self.params = [self.W_a, self.W_p, self.W]

    def forward(self, x, m):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_h
        :param m: 1D: batch, 2D: n_prds; prd index
        :return: 1D: n_words, 2D: batch * n_prds, 3D: dim_h
        """
        p = self.extract_prd_vectors(x, m)

        # 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        x = T.dot(x, self.W_a)
        x = x.dimshuffle(0, 'x', 1, 2)
        h_x = T.repeat(x, p.shape[1], axis=1)

        # 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        p = T.dot(p, self.W_p)
        p = p.dimshuffle(0, 1, 'x', 2)
        h_p = T.repeat(p, x.shape[2], axis=2)

        # 1D: batch, 2D: n_prds, 3D: n_words, 4D: dim_h
        h = T.dot(T.concatenate([h_x, h_p], axis=3), self.W)
        # 1D: batch * n_prds, 2D: n_words, 3D: dim_h
        h = h.reshape((h.shape[0] * h.shape[1], h.shape[2], h.shape[3]))
        return h.dimshuffle(1, 0, 2)

    @staticmethod
    def extract_prd_vectors(x, p):
        """
        :param x: 1D: batch, 2D: n_words, 3D: dim_h
        :param p: 1D: batch, 2D: n_prds; prd index
        :return: 1D: batch, 2D: n_prds, 3D: dim_h
        """
        v = T.arange(x.shape[0])
        v = T.repeat(v, p.shape[1], axis=0)
        return x[v, p.flatten()].reshape((p.shape[0], p.shape[1], x.shape[2]))


class RankingLayer(SoftmaxLayer):

    def __init__(self, n_i, n_labels):
        super(RankingLayer, self).__init__(n_i, n_labels)

    def forward(self, x):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels; log probability of a label
        """
        return T.dot(x, self.W)

    @staticmethod
    def get_y_scores(h, y):
        """
        :param h: 1D: n_words+1, 2D: batch, 3D: n_labels; score of a label
        :param y: 1D: batch, 2D: n_labels; word index
        :return: 1D: batch, 2D: n_labels; the highest scores
        """
        # 1D: batch, 2D: n_labels, 3D: n_words+1
        h = h.dimshuffle(1, 2, 0)
        # 1D: batch * n_labels, 2D: n_words+1
        h_reshaped = h.reshape((h.shape[0] * h.shape[1], -1))
        # 1D: batch, 2D: n_labels
        y = y.ravel()
        return h_reshaped[T.arange(y.shape[0]), y].reshape((h.shape[0], h.shape[1]))

    @staticmethod
    def get_y_hat_scores(h):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels; score of a label
        :return: 1D: batch, 2D: n_labels; the highest scores
        """
        return T.max(h, axis=0)

    @staticmethod
    def decode(h):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels; score of a label
        :return: 1D: batch, 2D: n_labels; the highest scoring argument index
        """
        return T.argmax(h, axis=0)


class MEMMLayer(object):

    def __init__(self, n_i, n_labels):
        self.W = theano.shared(sample_weights(n_i, n_labels))
        self.W_trans = theano.shared(sample_weights(n_labels, n_labels))
        self.params = [self.W, self.W_trans]

    def forward(self, x, h):
        """
        :param x: 1D: n_words, 2D: batch, 3D: dim_h
        :param h: 1D: n_words, 2D: batch, 3D: dim_h
        :return: 1D: n_words, 2D: batch, 3D: n_labels; log probability of a label
        """
        return relu(T.dot(T.concatenate([x, h], 2), self.W))

    def get_y_prob(self, h, y):
        """
        :param h: 1D: n_words * batch, 2D: n_labels
        :param y: 1D: n_words, 2D: batch
        :return: 1D: batch; log probability of the correct sequence
        """
        return get_path_prob_memm(h, y, self.W_trans)

    def decode(self, h):
        """
        :param h: 1D: n_words * batch, 2D: n_labels
        :return: 1D: batch, 2D: n_words; the highest scoring sequence (label id)
        """
        return greedy_search(h, self.W_trans).dimshuffle(1, 0)


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
        return get_path_prob_crf(h, y, self.W_trans)

    def decode(self, h):
        """
        :param h: 1D: n_words, 2D: batch, 3D: n_labels
        :return: 1D: batch, 2D: n_words; the highest scoring sequence (label id)
        """
        return viterbi_search(h, self.W_trans).dimshuffle(1, 0)
