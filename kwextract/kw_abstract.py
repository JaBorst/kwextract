import pandas as pd
import scipy as sp
from collections import Counter
import nltk

class KWAbstract:
    def __init__(self, ngrams=None, pos_pattern=[], cut_first=200, min_freq=3):
        self.ngrams = ngrams
        self.cut_first = cut_first
        self.min_freq = min_freq
        self.pos_pattern = pos_pattern

    def create_ngrams(self, l, add_ngrams):
        return ["_".join(n) for i in add_ngrams for n in nltk.ngrams(l, n=i) ]

    def create_patterns(self, l, pos_pattern):
        # todo
        lengths = list(set([len(x) for x in pos_pattern]))
        str_pats = ["_".join(x) for x in pos_pattern]
        pats = [ "_".join([x[1] for x in x]) for length in lengths for x in nltk.ngrams(l, n=length) if "_".join([x[2] for x in x]) in str_pats]
        return pats

    def create_dtm(self, texts):
        # Count texts
        self.term_frequencies = {group: Counter(text) for group, text in texts.items()}
        self.vocab = sum(self.term_frequencies.values(),Counter())

        # Prune overall vocabulary
        self.vocab = sorted(list(self.vocab.items()), key=lambda x: -x[1])
        self.vocab = [x[0] for x in self.vocab[self.cut_first:] if x[1] >= self.min_freq]

        # Count and restrict domain level text to the vocabulary
        self.term_frequencies = {group: Counter(text) for group, text in texts.items()}
        # self.dtm = np.array([[v.get(w, 0) for w in self.vocab] for k, v in self.term_frequencies.items()])
        _vocab_enum = {w:i for i,w in enumerate(self.vocab)}
        dat = [(n,i,_vocab_enum[w])    for i, v in enumerate(self.term_frequencies.values()) for w,n in v.items() if w in _vocab_enum]

        self.dtm = sp.sparse.coo_matrix(([x[0] for x in dat], ([x[1] for x in dat],[x[2] for x in dat],)), shape=(len(self.term_frequencies), len(self.vocab))).tocsr()

    def __call__(self,  df, groups, n=10):
        self.df = df
        self.groups = groups

        if self.ngrams is not None:
            ngrams = self.df.groupby(groups)["token"].apply(lambda x: self.create_ngrams(x.tolist(), add_ngrams=self.ngrams)).to_dict()
        if  len(self.pos_pattern) > 0:
            pos = self.df.groupby(groups)[["token","POS"]].apply(lambda x: self.create_patterns(list(x.to_records()), self.pos_pattern)).to_dict()

        self.texts = {k: list(v)[0] for k, v in
                      list(pd.DataFrame((self.df.groupby(self.groups)["token"].apply(list))).iterrows())}
        if self.ngrams is not None:
            self.texts = {k: self.texts[k] + ngrams[k] for k in self.texts.keys()}
        if len(self.pos_pattern) > 0:
            self.texts = {k: self.texts[k] + pos[k] for k in self.texts.keys()}

        self.create_dtm(self.texts)
        return self.keywords(n=n)