import theano
import theano.tensor as T
from utils import sample_weights


class AttentionLayer(object):

    def __init__(self, n_h):
        self.W1_c = theano.shared(sample_weights(n_h, n_h))
        self.W1_h = theano.shared(sample_weights(n_h, n_h))
        self.w    = theano.shared(sample_weights(n_h, ))
        self.W2_r = theano.shared(sample_weights(n_h, n_h))
        self.W2_h = theano.shared(sample_weights(n_h, n_h))
        self.params = [self.W1_c, self.W1_h, self.w, self.W2_r, self.W2_h]

    def seq_attention(self, h):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D n_h
        :return: y: 1D: n_words, 2D: batch_size, 3D: n_h
        """
        y, _ = theano.scan(fn=self.one_attention,
                           sequences=h,
                           outputs_info=None,
                           non_sequences=[h.dimshuffle(1, 0, 2),
                                          self.W1_c, self.W1_h, self.w, self.W2_r, self.W2_h]
                           )
        return y

    def multi_seq_attention(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size, 3D n_h
        :return: y: 1D: n_words, 2D: batch_size, 3D: n_h
        """
        # 1D: batch_size (n_sents * n_prds), 2D: n_words, 3D: n_h
        C = h.dimshuffle(1, 0, 2)
        # 1D: n_sents, 2D: n_words * n_prds, 3D: n_h
        C = C.reshape((h.shape[1] / n_prds, h.shape[0] * n_prds, h.shape[2]))
        # 1D: n_sents * n_prds, 2D: n_words * n_prds, 3D: n_h
        C = T.repeat(C, n_prds, 0)

        y, _ = theano.scan(fn=self.one_ms_attention,
                           sequences=h,
                           outputs_info=None,
                           non_sequences=[C, self.W1_c, self.W1_h, self.w, self.W2_r, self.W2_h]
                           )
        return y

    def multi_prd_attention(self, h, n_prds):
        """
        :param h: 1D: n_words, 2D: batch_size (n_sents * n_prds), 3D n_h
        :return: y: 1D: n_words, 2D: batch_size, 3D: n_h
        """
        # 1D: n_words, 2D: batch_size, 3D: n_prds, 4D: n_h
        C = h.reshape((h.shape[0], h.shape[1]/n_prds, n_prds, h.shape[2]))

        # 1D: n_words, 2D: batch_size * n_prds, 3D: n_prds, 4D: n_h
        C = T.repeat(C, n_prds, 1)

        y, _ = theano.scan(fn=self.one_mp_attention,
                           sequences=[h, C],
                           outputs_info=None,
                           non_sequences=[self.W1_c, self.W1_h, self.w, self.W2_r, self.W2_h]
                           )
        return y

    def one_mp_attention(self, h_t, C_t, W1_c, W1_h, w, W2_r, W2_h):
        """
        :param h_t: 1D: batch_size, 2D n_h
        :param C_t: 1D: batch_size * n_prds, 2D: n_prds, 3D n_h
        :return: 1D: batch_size, 2D: n_h
        """

        # 1D: batch_size, 2D: n_prds, 3D: n_h
        M = T.tanh(T.dot(C_t, W1_c) + T.dot(h_t, W1_h).dimshuffle(0, 'x', 1))

        # 1D: batch_size, 2D: n_prds, 3D: 1
        alpha = T.nnet.softmax(T.dot(M, w))
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch_size, 2D: n_h
        r = T.sum(C_t * alpha, axis=1)

        # 1D: batch_size, 2D: n_h
        return T.tanh(T.dot(r, W2_r) + T.dot(h_t, W2_h))

    def one_attention(self, h_t, C, W1_c, W1_h, w, W2_r, W2_h):
        """
        :param h_t: 1D: batch_size, 2D n_h
        :param C: 1D: batch_size, 2D: n_words, 3D n_h
        :return: 1D: batch_size, 2D: n_h
        """

        # 1D: batch_size, 2D: n_words, 3D: n_h
        M = T.tanh(T.dot(C, W1_c) + T.dot(h_t, W1_h).dimshuffle(0, 'x', 1))

        # 1D: batch_size, 2D: n_words, 3D: 1
        alpha = T.nnet.softmax(T.dot(M, w))
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch_size, 2D: n_h
        r = T.sum(C * alpha, axis=1)

        # 1D: batch_size, 2D: n_h
        return T.tanh(T.dot(r, W2_r) + T.dot(h_t, W2_h))

    def one_ms_attention(self, h_t, C, W1_c, W1_h, w, W2_r, W2_h):
        """
        :param h_t: 1D: batch_size, 2D n_h
        :param C: 1D: batch_size, 2D: n_words * n_prds, 3D n_h
        :return: 1D: batch_size, 2D: n_h
        """

        # 1D: batch_size, 2D: n_words * n_prds, 3D: n_h
        M = T.tanh(T.dot(C, W1_c) + T.dot(h_t, W1_h).dimshuffle(0, 'x', 1))

        # 1D: batch_size, 2D: n_words * n_prds
        alpha = T.nnet.softmax(T.dot(M, w))
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch_size, 2D: n_h
        r = T.sum(C * alpha, axis=1)

        # 1D: batch_size, 2D: n_h
        return T.tanh(T.dot(r, W2_r) + T.dot(h_t, W2_h))

