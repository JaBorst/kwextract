# kwextract - Keyword extraction 

Easy to use Keyword extractor using differential analysis.
Install with:

```bash
pip install git+
```

## Usage

```python
texts = [
    "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein gebürtiger deutscher Physiker mit Schweizer und US-amerikanischer Staatsbürgerschaft. Er gilt als einer der bedeutendsten theoretischen Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit.",
    "Niels Henrik David Bohr (* 7. Oktober 1885 in Kopenhagen; † 18. November 1962 ebenda) war ein dänischer Physiker. Er erhielt 1921 die Hughes-Medaille der Royal Society[1] und den Nobelpreis für Physik im Jahr 1922 „für seine Verdienste um die Erforschung der Struktur der Atome und der von ihnen ausgehenden Strahlung“.",
    "Werner Karl Heisenberg (* 5. Dezember 1901 in Würzburg; † 1. Februar 1976 in München) war ein deutscher Physiker. Heisenberg gab 1925 die erste mathematische Formulierung der Quantenmechanik an. 1927 formulierte er die Heisenbergsche Unschärferelation, die eine der fundamentalen Aussagen der Quantenmechanik trifft – nämlich, dass bestimmte Messgrößen eines Teilchens, etwa dessen Ort und dessen Impuls, nicht gleichzeitig beliebig genau zu bestimmen sind."
]

from kwextract import TFIDF
kw = TFIDF()
kw([x.split(" ") for x in texts], n=5)

>> [['Physiker', 'einer', 'als', 'US-amerikanischer', 'April'],
 ['die', 'den', 'im', 'erhielt', '1921'],
 ['Heisenberg', 'Quantenmechanik', 'dessen', 'die', 'sind.']]
```

When you want to extract keywords based on a reference corpus yu can do so by using 
loglikelihood method:

```python

texts = [
    "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein gebürtiger deutscher Physiker mit Schweizer und US-amerikanischer Staatsbürgerschaft. Er gilt als einer der bedeutendsten theoretischen Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit.",
    "Niels Henrik David Bohr (* 7. Oktober 1885 in Kopenhagen; † 18. November 1962 ebenda) war ein dänischer Physiker. Er erhielt 1921 die Hughes-Medaille der Royal Society[1] und den Nobelpreis für Physik im Jahr 1922 „für seine Verdienste um die Erforschung der Struktur der Atome und der von ihnen ausgehenden Strahlung“.",
    "Werner Karl Heisenberg (* 5. Dezember 1901 in Würzburg; † 1. Februar 1976 in München) war ein deutscher Physiker. Heisenberg gab 1925 die erste mathematische Formulierung der Quantenmechanik an. 1927 formulierte er die Heisenbergsche Unschärferelation, die eine der fundamentalen Aussagen der Quantenmechanik trifft – nämlich, dass bestimmte Messgrößen eines Teilchens, etwa dessen Ort und dessen Impuls, nicht gleichzeitig beliebig genau zu bestimmen sind.",
]

reference = "Marie Skłodowska Curie (* 7. November 1867 in Warschau, Russisches Kaiserreich; † 4. Juli 1934 bei Passy, geborene Maria Salomea Skłodowska) war eine Physikerin und Chemikerin polnischer Herkunft, die in Frankreich lebte und wirkte. Sie untersuchte die 1896 von Henri Becquerel beobachtete Strahlung von Uranverbindungen und prägte für diese das Wort „radioaktiv“. "

from kwextract import LLH
kw = LLH(reference=reference.split(" "))
kw.call_list([x.split(" ") for x in texts], n = 10)
```

These are just examples. Quality of extracted keywords improves with 
the amount of text you put in.


**_Disclaimer_** This repository is work in progress.

--
For reference see:

_Heyer, Gerhard, Quasthoff, Uwe and Wittig, Thomas. Text Mining: Wissensrohstoff Text: Konzepte, Algorithmen, Ergebnisse. 1., Aufl. : W3l, 2006._