if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='The Neural PAS System')

    parser.add_argument('-mode', default='raw', help='raw')

    parser.add_argument('--data', help='path to data')
    parser.add_argument('--out_data', help='path to output data', default='ntc.raw.txt')
    parser.add_argument('--out_vocab', help='path to output vocab', default='ntc.vocab.txt')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_cut_off', type=int, default=0)

    argv = parser.parse_args()
    print
    print argv
    print

    if argv.mode == 'raw':
        import gen_text
        gen_text.main(argv)

