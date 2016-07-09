import theano

theano.config.floatX = 'float32'


if __name__ == '__main__':
    import argparse
    import train

    parser = argparse.ArgumentParser(description='Train NN PAS System')

    """ Mode """
    parser.add_argument('-mode', default='train', help='train/test')

    """ Model """
    parser.add_argument('--model', default='char', help='word/char')
    parser.add_argument('--save', type=bool, default=True, help='save model')
    parser.add_argument('--load', type=str, default=None, help='load model')

    """ Data """
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--true_data', help='path to true data')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_size', type=int, default=100000000)

    """ Neural Architectures """
    parser.add_argument('--dim_emb',    type=int, default=32, help='dimension of embeddings')
    parser.add_argument('--dim_hidden', type=int, default=32, help='dimension of hidden layer')
    parser.add_argument('--unit', default='gru', help='unit')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--layer',  type=int, default=1,         help='number of layers')

    """ Training Parameters """
    parser.add_argument('--v_threshold', type=int, default=1, help='vocab threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='mini batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')

    args = parser.parse_args()

    train.main(args)
