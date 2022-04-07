import pandas as pd
import scipy as sp
from collections import Counter
import nltk

class KWAbstract:
    def __init__(self, ngrams=None,cut_first=0, min_freq=0):
        """
        Instantiate a keyword extractor, setting the pruning arguments.
        :param ngrams: Add ngrams to the list of tokens
        :param cut_first: Prune the first `cut_first`  most frequent tokens from the DTM (commonly referred to as stopwords) (default: 0)
        :param min_freq: Specify the minimum term frequency. All tokens rarer than that are also discarded.
        """
        self.ngrams = ngrams if hasattr(ngrams, '__iter__') or ngrams is None else [ngrams]
        self.cut_first = cut_first
        self.min_freq = min_freq

    def create_ngrams(self, l, add_ngrams):
        """Create ngrams from list"""
        return ["_".join(n) for i in add_ngrams for n in nltk.ngrams(l, n=i) ]

    def create_patterns(self, l, pos_pattern):
        """Not yet used"""
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
        _vocab_enum = {w:i for i,w in enumerate(self.vocab)}
        dat = [(n,i,_vocab_enum[w])    for i, v in enumerate(self.term_frequencies.values()) for w,n in v.items() if w in _vocab_enum]

        self.dtm = sp.sparse.coo_matrix(([x[0] for x in dat], ([x[1] for x in dat],[x[2] for x in dat],)), shape=(len(self.term_frequencies), len(self.vocab))).tocsr()

    def call_df(self,  texts: pd.DataFrame, groups=None, text="token", n=10):

        self.df = texts
        self.groups = groups
        self.texts = {k: v.to_list()[0] for k, v in list(pd.DataFrame((self.df.groupby(self.groups)["token"].apply(list))).iterrows())}
        r = self.call_dict(self.texts)
        return r

    def call_dict(self, texts: dict, n=10):
        self.texts = texts
        if self.ngrams is not None:
            self.texts = {k:v+self.create_ngrams(v, add_ngrams=self.ngrams) for k,v in self.texts.items()}
        self.create_dtm(self.texts)
        return self.keywords(n=n)


    def call_list(self, texts: list, n=10):
        r = self.call_dict(dict(enumerate(texts)), n=n)
        return list(r.values())

    def __call__(self, texts, *args, **kwargs):
        if isinstance(texts, list):
            return self.call_list(texts=texts, *args, **kwargs)
        elif isinstance(texts, dict):
            return self.call_dict(texts=texts, *args, **kwargs)
        elif isinstance(texts, pd.DataFrame):
            return self.call_df(texts=texts, *args, **kwargs)
        else:
            raise TypeError("KWExtractor only supports lists, dicts and Dataframes")