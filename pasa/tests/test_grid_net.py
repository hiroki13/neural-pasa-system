import numpy as np
import theano
import theano.tensor as T

np.random.seed(0)


def main():
    print test_grid_propagate()


def test_grid_forward():
    from ..nn.rnn import GridNetwork

    batch = 1
    n_prds = 2
    n_words = 3
    dim_h = 4
    grid_net = GridNetwork(unit='gru', depth=1, n_in=dim_h, n_h=dim_h)

    x = T.ftensor4()
    h = T.zeros(x.shape, dtype=theano.config.floatX)
    # 1D: batch, 2D: n_prds, 3D: dim_h
    h = grid_net.downward_all(grid_net.layers[0], x, h)

    f = theano.function(inputs=[x], outputs=h)

    x_in = np.ones(shape=(batch, n_prds, n_words, dim_h), dtype='float32')
    return f(x_in)


def test_grid_propagate():
    from ..nn.rnn import GridNetwork

    batch = 2
    n_prds = 2
    n_words = 3
    dim_h = 4
    grid_net = GridNetwork(unit='gru', depth=1, n_in=dim_h, n_h=dim_h)

    x = T.ftensor4()
    h = grid_net.forward(x)

    f = theano.function(inputs=[x], outputs=h)

    x_in = np.ones(shape=(batch, n_prds, n_words, dim_h), dtype='float32')
    return f(x_in)


if __name__ == '__main__':
    main()
