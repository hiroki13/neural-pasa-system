import theano

theano.config.floatX = 'float32'


if __name__ == '__main__':
    import argparse
    import stats

    parser = argparse.ArgumentParser(description='The Neural PAS System')

    parser.add_argument('--data', help='path to training data')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_size', type=int, default=1)
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')

    argv = parser.parse_args()

    stats.main(argv)
