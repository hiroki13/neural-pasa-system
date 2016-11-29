from heapq import heappush, heappop


class Graph(object):

    def __init__(self, matrix):
        self.nodes = self._gen_nodes(matrix)
        self.n_column = len(self.nodes[0])
        self.n_row = len(self.nodes)

    @staticmethod
    def _gen_nodes(matrix):
        len_column = len(matrix[0])
        len_row = len(matrix)
        prev_nodes = [[None for c in xrange(len_column)] for r in xrange(len_row)]
        BOS = Node(-1, -1, 0., [])
        for c in xrange(len_column):
            column = [prev_nodes[r][c-1] for r in xrange(len_row)]
            for r in xrange(len_row):
                if c > 0:
                    prev_nodes[r][c] = Node(r, c, matrix[r][c], column)
                else:
                    prev_nodes[r][c] = Node(r, c, matrix[r][c], [BOS])
        return prev_nodes

    def column(self, i):
        return [self.nodes[r][i] for r in xrange(self.n_row)]


class Node(object):

    def __init__(self, r_index, c_index, score, prev_nodes):
        self.r_index = r_index
        self.c_index = c_index
        self.score = score
        self.prev_nodes = prev_nodes
        self.next_node = None

        self.h = 0.
        self.g = 0.
        self.f = 0.

    def __le__(self, other):
        return self.f >= other.f

    def get_seq(self):
        seq = []
        node = self
        while node.c_index > -1:
            seq.append(node.r_index)
            node = node.next_node
        return seq

    def show(self):
        print 'r:%d\tc:%d\ts:%f\tf:%d\tg:%d\th:%d' % (self.r_index, self.c_index, self.score, self.h, self.g, self.f)


class PriorityQueue(object):

    def __init__(self):
        self.queue = []

    def push(self, value):
        heappush(self.queue, value)

    def pop(self):
        return heappop(self.queue)

    def empty(self):
        if self.__len__() == 0:
            return True
        return False

    def __len__(self):
        return len(self.queue)

    def __contains__(self, item):
        return item in self.queue

