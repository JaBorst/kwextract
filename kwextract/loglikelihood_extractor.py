import numpy as np
import scipy as sp
from .kw_abstract import KWAbstract

class LLH(KWAbstract):
    def __init__(self, threshold=None, **kwargs):
        super(LLH, self).__init__(**kwargs)
        self.threshold = threshold

    def _ll(self, a, b, n):

        c = a.sum()
        d = b.sum()

        e1 = (c * (a + b) / (c + d)).toarray() + np.array([1e-25])
        e2 = (d * (a + b) / (c + d)).toarray()  + np.array([1e-25])

        ll = 2 * ((a.toarray() * np.log(a.toarray() / e1 + 1e-25)) + b.toarray() * np.log(b.toarray() / e2 + 1e-25) )

        if self.threshold is not None:
            return [self.vocab[k.item()] for k in np.where(ll[0] > self.threshold)[0]]

        if n is not None:
            return [self.vocab[k] for k in ll[0].argsort(0)[::-1][:n]]

    def log_likelihood(self, k, n=None):
        i = list(self.texts.keys()).index(k)

        a = self.dtm[i]
        b = self.dtm[list(self.texts.keys()).index("reference")]
        return self._ll(a,b,n)

    def keywords(self, n=None):
        return {k: self.log_likelihood(k, n=n) for k in self.texts.keys() if k != "reference"}

