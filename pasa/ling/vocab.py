PAD = u'<PAD>'
UNK = u'<UNK>'
MARK = u'<MARK>'
NMARK = u'<NMARK>'

NA = u'NA'
GA = u'Ga'
O = u'O'
NI = u'Ni'
PRD = u'V'

GA_LABEL = 0
O_LABEL = 1
NI_LABEL = 2


class Vocab(object):

    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def set_init_word(self):
        self.add_word(PAD)
        self.add_word(MARK)
        self.add_word(NMARK)
        self.add_word(UNK)

    def set_pas_labels(self):
        self.add_word(NA)
        self.add_word(GA)
        self.add_word(O)
        self.add_word(NI)
        self.add_word(PRD)

    def set_pas_labels_char(self):
        bio = ['B-', 'I-']
        self.add_word(NA)
        for i in bio:
            self.add_word(i + GA)
        for i in bio:
            self.add_word(i + O)
        for i in bio:
            self.add_word(i + NI)
        for i in bio:
            self.add_word(i + PRD)

    def add_vocab(self, word_freqs, vocab_cut_off=0):
        for w, freq in sorted(word_freqs.items(), key=lambda (k, v): -v):
            if freq == vocab_cut_off:
                break
            self.add_word(w)

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab