from copy import deepcopy

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
            best_list = self.decode_argmax(prob_lists, prd_indices)
            n_best_list = self.decode_n_best_each(best_list=best_list, prob_lists=prob_lists, N=N)
            n_best_lists.append(n_best_list)
        return n_best_lists

    def decode_n_best_each(self, best_list, prob_lists, N):
        """
        :param best_list: 1D: n_prds, 2D: n_words; label id
        :param prob_lists: 1D: n_prds, 2D: n_words, 3D: n_labels: log probability
        :return: n_best_list: 1D: n_prds, 3D: n_words; label id
        """
        sorted_probs = self.sort_probs(best_list, prob_lists)
        n_best_list = [best_list]
        n_th_list = deepcopy(best_list)
        for n in xrange(N-1):
            if len(sorted_probs) == 0:
                return n_best_list
            n_th_list = self.create_n_th_list(n_th_list, sorted_probs)
            n_best_list.append(n_th_list)
        return n_best_list

    @staticmethod
    def sort_probs(best_list, prob_lists):
        """
        :param best_list: 1D: n_prds, 2D: n_words; label index
        :param prob_lists: 1D: n_prds, 2D: n_words, 3D: n_labels
        :return: sorted_list: 1D: n_words * n_labels; (prd_index, word index, label index, prob)
        """
        sorted_list = []
        for prd_index, prob_list in enumerate(prob_lists):
            for word_index, probs in enumerate(prob_list):
                best_label_index = best_list[prd_index][word_index]
                if best_label_index == 4:
                    continue
                best_prob = np.max(prob_lists[prd_index][word_index])
                for label_index, prob in enumerate(probs[:-1]):
                    if label_index != best_label_index:
                        diff = best_prob - prob
                        sorted_list.append((prd_index, word_index, label_index, diff))
        sorted_list.sort(key=lambda x: x[-1])
        return sorted_list

    @staticmethod
    def create_n_th_list(cur_list, sorted_prob_list):
        n_th_list = deepcopy(cur_list)
        prd_index, word_index, label_index, prob = sorted_prob_list.pop(0)
        n_th_list[prd_index][word_index] = label_index
        return n_th_list

    @staticmethod
    def forward_dp(matrix):
        """
        :param matrix: 1D: n_prds * n_words, 2D: n_labels; label probability
        :return: 1D: n_prds, 2D: n_words; label index
        """
        graph = Graph(matrix)
        for i in xrange(graph.n_column):
            nodes = graph.column(i)
            for node in nodes:
                score = -1000000
                best_prev = None
                for prev in node.prev_nodes:
                    tmp_score = prev.f + node.score
                    if tmp_score > score:
                        score = tmp_score
                        best_prev = prev
                node.best_prev_node = best_prev
                node.f = score
        return graph

    @staticmethod
    def backward_a_star(graph, n_best):
        result = PriorityQueue()
        q = PriorityQueue()
        EOS = Node(-1, -1, 0., graph.column(graph.n_column-1))
        q.push(EOS)

        while not q.empty():
            node = q.pop()
            if node.c_index == 0:
                result.push(node)
            else:
                for prev in node.prev_nodes:
                    """
                    prev.g = node.g + node.score
                    prev.next_node = node
                    prev.h = prev.f + prev.g
                    q.push(prev)
                    """
                    p = Node(prev.r_index, prev.c_index, prev.score, prev.prev_nodes)
                    p.f = prev.f
                    p.g = node.g + node.score
                    p.h = p.f + p.g
                    p.next_node = node
                    q.push(p)
            if len(result) == n_best:
                return result
