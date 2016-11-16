import numpy as np
import theano
import theano.tensor as T

np.random.seed(0)


def main():
    print test_flip()


def test_grid_forward():
    from ..nn.rnn import GridObliqueNetwork

    batch = 1
    n_prds = 2
    n_words = 3
    dim_h = 4
    grid_net = GridObliqueNetwork(unit='gru', depth=1, n_in=dim_h, n_h=dim_h)

    x = T.ftensor4()
    h = T.zeros(x.shape, dtype=theano.config.floatX)
    # 1D: batch, 2D: n_prds, 3D: dim_h
    h = grid_net.downward_all(grid_net.layers[0], x, h)

    f = theano.function(inputs=[x], outputs=h)

    x_in = np.ones(shape=(batch, n_prds, n_words, dim_h), dtype='float32')
    return f(x_in)


def test_grid_propagate():
    from ..nn.rnn import GridObliqueNetwork

    batch = 2
    n_prds = 2
    n_words = 3
    dim_h = 4
    grid_net = GridObliqueNetwork(unit='gru', depth=1, n_in=dim_h, n_h=dim_h)

    x = T.ftensor4()
    h = grid_net.forward(x)

    f = theano.function(inputs=[x], outputs=h)

    x_in = np.ones(shape=(batch, n_prds, n_words, dim_h), dtype='float32')
    return f(x_in)


def test_flip():
    from ..nn.rnn import GridObliqueNetwork

    x = T.fmatrix()
    dim_h = 4
    g_net = GridObliqueNetwork(unit='gru', depth=1, n_in=dim_h, n_h=dim_h)
    h = g_net.flip(x)
    f = theano.function(inputs=[x], outputs=h)

    x_in = np.asarray([[i + j for i in xrange(3)] for j in xrange(4)], dtype='float32')
    print x_in
    return f(x_in)


if __name__ == '__main__':
    main()
