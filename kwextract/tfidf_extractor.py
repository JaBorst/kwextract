import numpy as np
from .kw_abstract import KWAbstract

class TFIDF(KWAbstract):
    def tfidf(self):
        tf = np.log(self.dtm + 1e-25)
        idf = np.log(self.dtm.shape[0] / ((self.dtm > 0).sum(0) + 1e-25))
        return tf * idf

    def keywords(self, n=10):
        """Iterate all copora for printing"""
        tfidf = self.tfidf()
        return {k: [self.vocab[k] for k in tfidf[i].argsort(0)[::-1][:n]] for i, k in enumerate(self.texts.keys())}


