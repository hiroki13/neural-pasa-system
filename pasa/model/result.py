class Results(object):

    def __init__(self, argv):
        self.argv = argv
        self.samples = []
        self.outputs_prob = []
        self.outputs_hidden = []
        self.decoder_outputs = []

    def add(self, elems):
        self.samples.append(elems[0])
        self.outputs_prob.append(elems[1][0])
        if self.argv.output == 'pretrain':
            self.outputs_hidden.append(elems[1][1])
        self.decoder_outputs.append(elems[2])
