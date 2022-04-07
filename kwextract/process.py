import pandas as pd
import spacy
import re

class SpacyProcess:
    def __init__(self):
        self. nlp = spacy.load("de_core_news_sm",
                     disable=['tok2vec', 'ner', 'attribute_ruler', 'lemmatizer', 'morphologizer', ])

        # self.nlp.add_pipe('sentencizer')


    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process_dict(self, x) -> pd.DataFrame:
        df = pd.DataFrame(x.items())
        df.columns = ["key", "text"]
        df.text = df.text.map(self.process_list)
        df = df.explode("text")
        df[["word_id", "token", "POS"]] = df.text.to_list()
        del df["text"]
        return df

    def process_list(self, x) -> pd.DataFrame:
        x = [x] if isinstance(x, str) else x
        x = sum([re.split("[\r\n]{2,}", d) for d in x],[]) # heuristcal split of sentences
        d = sum([self.process_str(d) for d in x],[])
        d = [tuple([i] + list(t)) for i,s in enumerate(d) for t in s]
        return d

    def process_str(self, x):
        return [[(tok.text, tok.pos_)for tok in sent] for sent in self.nlp(x.strip()).sents ]

    def process(self, x):
        x = [x] if isinstance(x, str) else x
        if isinstance(x, list) or isinstance(x, str):
            return self.process_list(x)
        if isinstance(x, dict):
            return self.process_dict(x)