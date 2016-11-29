def main():
    graph = test_forward_dp()
    _show_graph(graph)
    nodes = test_backward_a_star()
    _show_nodes(nodes)


def test_forward_dp():
    from ..decoder.decoder import Decoder
    import numpy as np
    np.random.seed(0)

    decoder = Decoder()
    row = 2
    column = 3
    matrix = [[np.random.randint(-10, 10) for c in xrange(column)] for r in xrange(row)]
    return decoder.forward_dp(matrix)


def test_backward_a_star():
    from ..decoder.decoder import Decoder
    decoder = Decoder()
    graph = test_forward_dp()
    return decoder.backward_a_star(graph, 8)


def test_gen_graph():
    from decoder.graph import Graph
    row = 3
    column = 4
    matrix = [[r + c for c in xrange(column)] for r in xrange(row)]
    graph = Graph(matrix)
    _show_graph(graph)


def _show_graph(graph):
    for c in xrange(len(graph.nodes[0])):
        for r in xrange(len(graph.nodes)):
            node = graph.nodes[r][c]
            print '%d %d %d %d' % (node.r_index, node.c_index, node.score, node.f)


def _show_p_queue(q):
    while not q.empty():
        node = q.pop()
        print '%d %d %d %d' % (node.r_index, node.c_index, node.score, node.f)


def _show_nodes(nodes):
    for node in nodes:
        print '%d\t%d\t%d\t%d\t%d\t%d' % (node.r_index, node.c_index, node.score, node.f, node.g, node.h)
        while node.next_node is not None:
            node = node.next_node
            print '%d\t%d\t%d\t%d\t%d\t%d' % (node.r_index, node.c_index, node.score, node.f, node.g, node.h)
        print


if __name__ == '__main__':
    main()
