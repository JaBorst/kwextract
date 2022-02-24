
#
# path = "/home/jb/keywords/Richtlinien für Textanalyse.zip"
# def read_content(z, pdf):
#     with tempfile.TemporaryDirectory() as dir:
#         z.extract(pdf, path=f"{dir}/")
#         with pdfplumber.open(f"{dir}/{pdf}") as f:
#             content = [x.extract_text() for x in f.pages]
#     return " ".join(content)
# content = []
# with zipfile.ZipFile(path) as zf:
#     for fname in tqdm(zf.namelist()):
#         try:
#             content.append((fname,read_content(zf, fname)))
#         except:
#             pass
#
# df = pd.DataFrame(content, columns=["file", "text"])
#
# def get_sections(x):
#     try:
#         x_split = re.split(r"\n[Teil]{0,4}§? ?[A-Z0-9]{1,5}.? ?\n(.{4,25})\n", x) #|
#         x_split = sum([re.split(r"\n[1-9IV]{1,2}.? ([a-zA-ZäöüÖÄÜ, ]{1,50})\n", t) for t in x_split],[])
#         paragraphs = [("Titel", x_split[0])]
#         paragraphs.extend([(x_split[i], x_split[i+1])for i in range(1,len(x_split),2)])
#         return pd.DataFrame(paragraphs)
#     except:
#         pass
#
# df = df.groupby("file").apply(lambda x:get_sections(x["text"].to_list()[0]))  # Problems 16,17,21  #|(\n[1-9]* .*\n) one line!
# df = df.reset_index()
import re
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
content["reference"] = reference[:10000]

from kwextract.process import SpacyProcess
df = SpacyProcess()(content)


from kwextract import TFIDF, LLH
# TFIDF()(df, groups=["file"])
kw = LLH( pos_pattern=[["DET", "ADJ", "Noun"], ["VERB", "ADV"], ["NOUN","PROPN", "NOUN"], ["NOUN", "NOUN"]])
kw(df, groups=["key"],n=100)
