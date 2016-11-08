def main(argv):
    print test_setup_training(argv)


def test_setup_training(argv):
    from ..api.train import RankingTrainer

    # 1D: n_sents (2), 2D: n_prds (2), 3D: n_words (2), 4D: dim_h (2)
    h = [
         [
          [[1, 2], [3, 4]],
          [[4, 3], [2, 1]]
         ],
         [
          [[3, 4], [1, 2]],
          [[2, 1], [4, 3]]
         ]
        ]
    # 1D: n_sents (2), 2D: n_prds (2), 3D: n_words (2), 4D: n_labels (3)
    p = [
         [
          [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
          [[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]
         ],
         [
          [[0.6, 0.2, 0.2], [0.1, 0.7, 0.2]],
          [[0.4, 0.2, 0.4], [0.5, 0.2, 0.3]]
         ]
        ]
    corpus_set = [(h, p), (None, None), (None, None)]

    trainer = RankingTrainer(argv=argv)
    trainer.corpus_set = corpus_set
    trainer.setup_training()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='The Neural PAS System')

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

    parser.add_argument('--batch_size', type=int, default=32, help='mini batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')

    argv = parser.parse_args()
    print
    print argv
    print

    main(argv)
