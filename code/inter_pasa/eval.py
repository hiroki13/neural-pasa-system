import numpy as np

GA_ID = 1
O_ID = 2
NI_ID = 3


class Eval(object):

    def __init__(self):
        self.corrects = 0.
        self.total = 0.
        self.nll = 0.

        self.accuracy = 0.

    def update_results(self, results):
        self.corrects += np.sum(results)
        self.total += len(results)

    def set_metrics(self):
        self. accuracy = self.corrects / self.total

    def show_results(self):
        self.set_metrics()
        print '\n\tEVALUATION'

        print '\t\tNLL: %f' % self.nll
        print '\n\tACCURACY: %f' % self.accuracy

