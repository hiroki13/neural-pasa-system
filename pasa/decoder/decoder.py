import numpy as np


class Decoder(object):

    def __init__(self, argv):
        self.argv = argv

    def decode(self, output_prob, prd_indices):
        """
        :param output_prob: 1D: n_prds, 2D: n_words, 3D: n_labels; label probability
        :param prd_indices: 1D: n_prds; prd word index in a sentence
        :return: 1D: n_prds, 2D: n_words; label index
        """
        best_lists = []
        for prob_list, prd_index in zip(output_prob, prd_indices):
            best_lists.append(self._decode_argmax(prob_list, prd_index))
        return best_lists

    @staticmethod
    def _decode_argmax(prob_list, prd_index):
        best_list = []
        for word_index, probs in enumerate(prob_list):
            if word_index == prd_index:
                best_label_index = 4
            else:
                best_label_index = np.argmax(probs[:-1])
            best_list.append(best_label_index)
        return best_list
