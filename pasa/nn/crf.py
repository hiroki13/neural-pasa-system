import theano
import theano.tensor as T

from utils import logsumexp, sample_weights


class Layer(object):

    def __init__(self, n_i, n_h):
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.params = [self.W]

        self.decode = self.memm

    def y_prob(self, h, y):
        """
        :param h: 1D: n_words, 2D: Batch, 3D: n_y
        :param y: 1D: n_words, 2D: Batch
        :return: gradient of cross entropy: 1D: Batch
        """

        # 1D: n_words * batch, 2D: n_y
        h_reshaped = h.reshape((h.shape[0] * h.shape[1], h.shape[2]))
        p_y = T.nnet.softmax(h_reshaped)
        # 1D: n_words * batch
        y = y.flatten()
        p_y_gold = T.log(p_y[T.arange(y.shape[0]), y])
        # 1D: batch, 2D: n_words
        p_y_gold = p_y_gold.reshape((h.shape[0], h.shape[1])).dimshuffle((1, 0))
        p_y_gold = T.sum(p_y_gold, axis=1)

        return p_y_gold

    def memm(self, h):
        # 1D: n_words * batch, 2D: n_y
        h_reshaped = h.reshape((h.shape[0] * h.shape[1], h.shape[2]))
        p_y = T.nnet.softmax(h_reshaped)
        # 1D: n_words * batch
        p_y_hat = T.argmax(p_y, axis=1)
        # 1D: batch, 2D: n_words
        p_y_hat = p_y_hat.reshape((h.shape[0], h.shape[1])).dimshuffle((1, 0))
        return p_y_hat


class CRFLayer(object):

    def __init__(self, n_i, n_h):
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.W_trans = theano.shared(sample_weights(n_h, n_h))
        self.BOS = theano.shared(sample_weights(n_h))
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
        :param scores_prev: 1D: Batch, 2D: n_y
        :param W_trans: 1D: n_y, 2D, n_y
        """
        score = score_prev.dimshuffle(0, 'x', 1) + W_trans + h_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_nodes_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, T.cast(max_nodes_t, dtype='int32')

    def backward(self, nodes_t, max_node_t):
        return nodes_t[T.arange(nodes_t.shape[0]), max_node_t]
