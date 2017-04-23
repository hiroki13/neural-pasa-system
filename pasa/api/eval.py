from ..utils.io_utils import say, CONLLLoader
from ..experimenter.evaluator import ResultEval


def main(argv):
    say('\n\nEVALUATING RESULTS\n')
    corpus_loader = CONLLLoader(min_unit='word', data_size=argv.data_size)
    evaluator = ResultEval()

    corpus = corpus_loader.load_corpus(argv.data)
    n_sents = len(corpus)
    n_prds = 0.
    for sent in corpus:
        n_prds += sent.size_prds()
    say('\tSent: %d\tPrds: %d\tPrds/Sent: %f\n' % (n_sents, n_prds, n_prds/n_sents))

    evaluator.calc_results(corpus)


