import numpy as np
import theano
import theano.tensor as T
import random

from ..intra_pasa.train import RerankingTrainer
from ..utils.preprocessor import RerankingPreprocessor

theano.config.floatX = 'float32'
np.random.seed(0)
random.seed(0)


def main(_argv):
    trainer = RerankingTrainer(_argv, RerankingPreprocessor(_argv))
    trainer.setup_training()
    print trainer.train_samples[0][-1]
    exit()
    test_grid_propagate(trainer.train_samples[0])


def test_grid_propagate(x_in):
    from ..nn.rnn import GridNetwork

    dim_h = 4
    grid_net = GridNetwork(unit='gru', depth=1, n_in=dim_h, n_h=dim_h)

    x = T.ftensor4()
    h = grid_net.forward(x)

    f = theano.function(inputs=[x], outputs=h)

    return f(x_in)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='The Neural PAS System')

    ########
    # Mode #
    ########
    parser.add_argument('-mode', default='train', help='train/test')

    #########
    # Model #
    #########
    parser.add_argument('--model', type=str, default='base', help='base/rank')
    parser.add_argument('--check', type=bool, default=False, help='check')
    parser.add_argument('--save', type=bool, default=False, help='save model')
    parser.add_argument('--result', type=bool, default=False, help='output results')
    parser.add_argument('--load_params', type=str, default=None, help='load trained parameters')
    parser.add_argument('--load_config', type=str, default=None, help='load configuration')

    ########
    # Data #
    ########
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_cut_off', type=int, default=0)
    parser.add_argument('--word', type=str, default=None, help='word')
    parser.add_argument('--label', type=str, default=None, help='label')

    ########################
    # Neural Architectures #
    ########################
    parser.add_argument('--unit', default='gru', help='unit')
    parser.add_argument('--fix', type=int, default=0, help='fix or not init embeddings')
    parser.add_argument('--layers',  type=int, default=1, help='number of layers')
    parser.add_argument('--output_layer',  type=int, default=0, help='softmax/memm/crf')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--dim_emb',    type=int, default=32, help='dimension of word embeddings')
    parser.add_argument('--dim_posit',  type=int, default=5, help='dimension of position embeddings')
    parser.add_argument('--dim_hidden', type=int, default=32, help='dimension of hidden layer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout prob')
    parser.add_argument('--attention', type=int, default=0, help='attention')
    parser.add_argument('--pooling', default='max', help='pooling')

    #######################
    # Training Parameters #
    #######################
    parser.add_argument('--batch_size', type=int, default=32, help='mini batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--n_best', type=int, default=5, help='How many best lists are created')
    parser.add_argument('--target', type=int, default=0, help='Jack Knife')

    argv = parser.parse_args()
    print
    print argv
    print

    main(argv)
