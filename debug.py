
import zipfile
content = {}
with zipfile.ZipFile("/home/jb/Downloads/keyword.zip") as zfile:
    for f in zfile.namelist():
        if f != "keyword/":
            content[f] = zfile.read(f).decode("utf8")
content.keys()
path = "/home/jb/Downloads/deu_news_2020_1M-sentences.txt"
with open(path, "r") as f:
    reference = [" ".join(x.split("\t")[1:]).strip() for x in  f.readlines()]
content["reference"] = " ".join(reference[:100])

content = {k:v.split(" ") for k, v in content.items()}

# from kwextract.process import SpacyProcess
# df = SpacyProcess()(content)


from kwextract import TFIDF, LLH
# TFIDF()(df, groups=["file"])
# kw = LLH( pos_pattern=[["DET", "ADJ", "Noun"], ["VERB", "ADV"], ["NOUN","PROPN", "NOUN"], ["NOUN", "NOUN"]])
# kw(df, groups=["key"],n=100)

kw = LLH()
kw.call_dict(content, 10)