from copy import deepcopy, copy

import numpy as np

from ..ling.graph import Graph, Node, PriorityQueue


class Decoder(object):

    def __init__(self):
        pass

    def decode_argmax(self, prob_lists, prd_indices):
        """
        :param prob_lists: 1D: n_prds, 2D: n_words, 3D: n_labels; label probability
        :param prd_indices: 1D: n_prds; prd word index in a sentence
        :return: 1D: n_prds, 2D: n_words; label index
        """
        best_lists = []
        for prob_list, prd_index in zip(prob_lists, prd_indices):
            best_lists.append(self.decode_argmax_each(prob_list, prd_index))
        return best_lists

    @staticmethod
    def decode_argmax_each(prob_list, prd_index):
        best_list = []
        for word_index, probs in enumerate(prob_list):
            if word_index == prd_index:
                best_label_index = 4
            else:
                best_label_index = np.argmax(probs[:-1])
            best_list.append(best_label_index)
        return best_list

    def decode_n_best(self, all_prob_lists, all_prd_indices, N):
        n_best_lists = []
        for prob_lists, prd_indices in zip(all_prob_lists, all_prd_indices):
            n_best_lists.append(self.decode_n_best_each(prob_lists, N))
        return n_best_lists

    def decode_n_best_each(self, prob_lists, N):
        """
        :param prob_lists: 1D: n_prds, 2D: n_words, 3D: n_labels
        :return: sorted_list: 1D: n_words * n_labels; (prd_index, word index, label index, prob)
        """
        n_prds = len(prob_lists)
        if n_prds == 0:
            return []
        matrix = self.flatten(prob_lists)
        nodes = self.backward_a_star(self.forward_dp(matrix), N)
        return self.get_n_best_list(nodes, n_prds)

    @staticmethod
    def forward_dp(matrix):
        """
        :param matrix: 1D: n_prds * n_words, 2D: n_labels; label probability
        :return: graph
        """
        graph = Graph(matrix)
        for i in xrange(graph.n_column):
            nodes = graph.column(i)
            for node in nodes:
                best_score = -1000000.
                for prev in node.prev_nodes:
                    tmp_score = prev.h + node.score
                    if tmp_score > best_score:
                        best_score = tmp_score
                node.h = best_score
        return graph

    @staticmethod
    def backward_a_star(graph, n_best):
        result = []
        q = PriorityQueue()
        EOS = Node(-1, -1, 0., graph.column(graph.n_column-1))
        q.push(EOS)

        while not q.empty():
            node = q.pop()
            if node.c_index == 0:
                result.append(node)
            else:
                for prev in node.prev_nodes:
                    p = copy(prev)
                    p.g = node.g + node.score
                    p.next_node = node
                    p.f = p.h + p.g
                    q.push(p)
            if len(result) == n_best:
                return result
        return result

    @staticmethod
    def get_n_best_list(nodes, n_prds):
        lists = []
        for node in nodes:
            matrix = []
            seq = node.get_seq()
            n_words = len(seq) / n_prds
            row = []
            for s in seq:
                row.append(s)
                if len(row) == n_words:
                    matrix.append(row)
                    row = []
            lists.append(matrix)
        return lists

    @staticmethod
    def flatten(tensor):
        tensor = np.asarray(tensor)
        matrix = tensor.reshape((tensor.shape[0] * tensor.shape[1], tensor.shape[2]))
        matrix = matrix.T
        return matrix

