from collections import Counter

UPOS = 3
XPOS = 4

def read_conllu(chemin, colonne=UPOS):
    phrases, etiquettes = [], []
    mots, tags = [], []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                if mots:
                    phrases.append(mots)
                    etiquettes.append(tags)
                    mots, tags = [], []
                continue
            if ligne.startswith("#"):
                continue
            cols = ligne.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            mots.append(cols[1])
            tags.append(cols[colonne])
    if mots:
        phrases.append(mots)
        etiquettes.append(tags)
    return phrases, etiquettes


def compter_etiquettes(etiquettes):
    return Counter(tag for tags in etiquettes for tag in tags)
