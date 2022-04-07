from kwextract import LLH

def test_llh_list():
    texts = [
        "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein gebürtiger deutscher Physiker mit Schweizer und US-amerikanischer Staatsbürgerschaft. Er gilt als einer der bedeutendsten theoretischen Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit.",
        "Niels Henrik David Bohr (* 7. Oktober 1885 in Kopenhagen; † 18. November 1962 ebenda) war ein dänischer Physiker. Er erhielt 1921 die Hughes-Medaille der Royal Society[1] und den Nobelpreis für Physik im Jahr 1922 „für seine Verdienste um die Erforschung der Struktur der Atome und der von ihnen ausgehenden Strahlung“.",
        "Werner Karl Heisenberg (* 5. Dezember 1901 in Würzburg; † 1. Februar 1976 in München) war ein deutscher Physiker. Heisenberg gab 1925 die erste mathematische Formulierung der Quantenmechanik an. 1927 formulierte er die Heisenbergsche Unschärferelation, die eine der fundamentalen Aussagen der Quantenmechanik trifft – nämlich, dass bestimmte Messgrößen eines Teilchens, etwa dessen Ort und dessen Impuls, nicht gleichzeitig beliebig genau zu bestimmen sind.",
    ]


    kw = LLH(reference=reference.split(" "))
    kw.call_list([x.split(" ") for x in texts], n = 10)

def test_llh_dict():
    texts = {
        "Einstein": "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein gebürtiger deutscher Physiker mit Schweizer und US-amerikanischer Staatsbürgerschaft. Er gilt als einer der bedeutendsten theoretischen Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit.",
        "Bohr": "Niels Henrik David Bohr (* 7. Oktober 1885 in Kopenhagen; † 18. November 1962 ebenda) war ein dänischer Physiker. Er erhielt 1921 die Hughes-Medaille der Royal Society[1] und den Nobelpreis für Physik im Jahr 1922 „für seine Verdienste um die Erforschung der Struktur der Atome und der von ihnen ausgehenden Strahlung“.",
        "Heisenberg": "Werner Karl Heisenberg (* 5. Dezember 1901 in Würzburg; † 1. Februar 1976 in München) war ein deutscher Physiker. Heisenberg gab 1925 die erste mathematische Formulierung der Quantenmechanik an. 1927 formulierte er die Heisenbergsche Unschärferelation, die eine der fundamentalen Aussagen der Quantenmechanik trifft – nämlich, dass bestimmte Messgrößen eines Teilchens, etwa dessen Ort und dessen Impuls, nicht gleichzeitig beliebig genau zu bestimmen sind.",
    }
    reference = "Marie Skłodowska Curie (* 7. November 1867 in Warschau, Russisches Kaiserreich; † 4. Juli 1934 bei Passy, geborene Maria Salomea Skłodowska) war eine Physikerin und Chemikerin polnischer Herkunft, die in Frankreich lebte und wirkte. Sie untersuchte die 1896 von Henri Becquerel beobachtete Strahlung von Uranverbindungen und prägte für diese das Wort „radioaktiv“. "

    kw = LLH(reference = reference.split(" "))
    kw.call_dict({k:v.split(" ") for k,v in texts.items()}, n = 10)

def test_llh_df():
    texts = {
        "Einstein": "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein gebürtiger deutscher Physiker mit Schweizer und US-amerikanischer Staatsbürgerschaft. Er gilt als einer der bedeutendsten theoretischen Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit.",
        "Bohr": "Niels Henrik David Bohr (* 7. Oktober 1885 in Kopenhagen; † 18. November 1962 ebenda) war ein dänischer Physiker. Er erhielt 1921 die Hughes-Medaille der Royal Society[1] und den Nobelpreis für Physik im Jahr 1922 „für seine Verdienste um die Erforschung der Struktur der Atome und der von ihnen ausgehenden Strahlung“.",
        "Heisenberg": "Werner Karl Heisenberg (* 5. Dezember 1901 in Würzburg; † 1. Februar 1976 in München) war ein deutscher Physiker. Heisenberg gab 1925 die erste mathematische Formulierung der Quantenmechanik an. 1927 formulierte er die Heisenbergsche Unschärferelation, die eine der fundamentalen Aussagen der Quantenmechanik trifft – nämlich, dass bestimmte Messgrößen eines Teilchens, etwa dessen Ort und dessen Impuls, nicht gleichzeitig beliebig genau zu bestimmen sind.",
    }
    reference = "Marie Skłodowska Curie (* 7. November 1867 in Warschau, Russisches Kaiserreich; † 4. Juli 1934 bei Passy, geborene Maria Salomea Skłodowska) war eine Physikerin und Chemikerin polnischer Herkunft, die in Frankreich lebte und wirkte. Sie untersuchte die 1896 von Henri Becquerel beobachtete Strahlung von Uranverbindungen und prägte für diese das Wort „radioaktiv“. "


    import pandas as pd
    df = pd.DataFrame({(k,t) for k,v in texts.items() for t in  v.split(" ") }, columns=["name", "token"])
    kw = LLH(reference = reference.split(" "))
    kw.call_df(df, groups=["name"], n = 10)
