import numpy as np
from collections import Counter
import nltk

class TFIDF:
    def __init__(self, df, groups):
        self.df = df
        self.groups = groups

        self.texts = {k: list(v)[0] for k, v in list(pd.DataFrame((self.df.groupby(self.groups)["text"].apply(list))).iterrows())}


    def create_ngrams(self, l, add_ngrams):
        return ["_".join(n) for i in add_ngrams for n in nltk.ngrams(l, n=i) ]

    def create_dtm(self, cut_first=200, min_freq=3, add_ngrams=None):

        # Count texts
        if add_ngrams is not None:
            self.texts = {k:v+self.create_ngrams(v,add_ngrams) for k, v in self.texts.items()}
        self.vocab = Counter(sum(self.texts.values(),[]))

        # Prune overall vocabulary
        self.vocab = sorted(list(self.vocab.items()), key=lambda x: -x[1])
        self.vocab = [x[0] for x in self.vocab[cut_first:] if x[1] >= min_freq]

        # count and restrict domain level text to the vocabulary
        self.term_frequencies = {group: Counter(text) for group, text in self.texts.items()}
        self.dtm = np.array([[v.get(w, 0) for w in self.vocab] for k, v in self.term_frequencies.items()])

    def tfidf(self):
        tf = np.log(self.dtm + 1e-25)
        # tf = 0.5 + 0.5 * self.dtm / self.dtm.max(-1, keepdims=True) # alternative normalization.
        idf = np.log(self.dtm.shape[0] / ((self.dtm > 0).sum(0) + 1e-25))
        return tf * idf

    def tfidf_keywords(self, n=10):
        """Iterate all copora for printing"""
        tfidf = self.tfidf()
        return {k: [self.vocab[k] for k in tfidf[i].argsort(0)[::-1][:n]] for i, k in enumerate(self.texts.keys())}


from tqdm import tqdm
import pdfplumber
import tempfile
import spacy
import zipfile
import pandas as pd
import re

path = "/home/jb/keywords/Richtlinien für Textanalyse.zip"
def read_content(z, pdf):
    with tempfile.TemporaryDirectory() as dir:
        z.extract(pdf, path=f"{dir}/")
        with pdfplumber.open(f"{dir}/{pdf}") as f:
            content = [x.extract_text() for x in f.pages]
    return " ".join(content)
content = []
with zipfile.ZipFile(path) as zf:
    for fname in tqdm(zf.namelist()):
        try:
            content.append((fname,read_content(zf, fname)))
        except:
            pass

df = pd.DataFrame(content, columns=["file", "text"])

def get_sections(x):
    try:
        x_split = re.split(r"\n[Teil]{0,4}§? ?[A-Z0-9]{1,5}.? ?\n(.{4,25})\n", x) #|
        x_split = sum([re.split(r"\n[1-9IV]{1,2}.? ([a-zA-ZäöüÖÄÜ, ]{1,50})\n", t) for t in x_split],[])
        paragraphs = [("Titel", x_split[0])]
        paragraphs.extend([(x_split[i], x_split[i+1])for i in range(1,len(x_split),2)])
        return pd.DataFrame(paragraphs)
    except:
        pass

df = df.groupby("file").apply(lambda x:get_sections(x["text"].to_list()[0]))  # Problems 16,17,21  #|(\n[1-9]* .*\n) one line!
df = df.reset_index()

df.columns = ["file", "-", "paragraph", "text"]
del df["-"]
df["text"] = df["text"].str.split()
df = df.explode("text")

kw = TFIDF(df=df, groups=["file"])
kw.create_dtm(add_ngrams=[2,3])
kw.tfidf_keywords(100)