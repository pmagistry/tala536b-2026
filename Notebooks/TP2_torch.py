import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP2 : classification en 6 Categories  (version PyTorch)

    L'objectif est de tester le changement de loss, l'activation softmax et de produire un classifieur pour plus de deux langues.

    ## Imports usuels

    (ici j'utilise un fichier python annexe pour une partie de mon code, pensez à aller le voir aussi)
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import torch
    from collections import Counter
    import re
    from sklearn.model_selection import train_test_split
    import utils_tp2_torch as u

    return mo, np, torch, train_test_split, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lecture des données

    (fonctions issues du TP précédent, déplacées dans la bibliothèque)

    Ce code peut aussi marcher avec le corpus6.txt

    Question:
    - est-il possible de critiquer la normalisation du corpus à cet endroit ?
    """)
    return


@app.cell
def _(u):
    dataset = u.load_json("./corpus/corpus-classif.json")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage et encodage du corpus

    Le découpage 80/20 est réalisé avec scikit-learn

    L'encodage se fait ici en ***sparse***, une version **one-hot** des fonctions et du modèle est disponible
    """)
    return


@app.cell
def _():
    classes = {'culture': 0, 'économie': 1, 'economie': 1, 'science': 2,
                   'international': 3, 'société': 4, 'societe': 4,
                   'sport': 5, 'politique': 6}
    #Encode les categories presente dans le corpus
    return (classes,)


@app.cell
def _(classes, dataset, np, train_test_split):
    textes = [d['text'] for d in dataset]
    y = np.array([classes[d['label']] for d in dataset], dtype=np.int64)

    textes_train, textes_test, Y_train, Y_test = train_test_split( textes, y, test_size=0.3, random_state=42, stratify=y)
    return Y_test, Y_train, textes_test, textes_train


@app.cell
def _(Y_test, Y_train, textes_test, textes_train, torch, u):
    vocab = u.construire_vocab(textes_train, max_tokens=5000)

    #vectorisation : 
    X_train = u.vectoriser(textes_train, vocab, longueur=500)
    X_test  = u.vectoriser(textes_test, vocab, longueur=500)


    torch.manual_seed(42) 
    embedding, tete = u.creer_modele_(vocab, nb_classes=7, dim=32,dropout=0.3)
    #embedding c'est une representation sementique des mots tranformé en vecturs, et la tete c'est ces vecteur auxquels ont applique des couches lineaires afin de pouvoir les classer dans l'une des categories.
    u.entrainer_(embedding, tete, X_train, Y_train, X_test, Y_test, pooling="avg", epochs=30, lr=1e-2)
    return X_test, X_train, embedding, tete, vocab


@app.cell
def _(X_test, X_train, Y_test, Y_train, embedding, tete, torch, u, vocab):
    torch.manual_seed(42)
    embedding_, tete_ = u.creer_modele_(vocab, nb_classes=7, dim=32,dropout=0.5)
    u.entrainer_(embedding, tete, X_train, Y_train, X_test, Y_test, pooling="max", epochs=100, lr=1e-2)
    return


@app.cell
def _(X_test, X_train, Y_test, Y_train, embedding, tete, u):
    #afin d'evaluer la robustesse du model en affichant la moyenne du train et teste ainsi que la valeur maximal de chaque.
    for p in ["avg", "max"]:
        a_tr = u.accuracy_(embedding, tete, X_train, Y_train, pooling=p)
        a_te = u.accuracy_(embedding, tete, X_test, Y_test, pooling=p)
        print(p, "train:", round(a_tr, 3), "test:", round(a_te, 3))
    return


@app.cell
def _(X_test, X_train, Y_test, Y_train, embedding, tete, u):
    #memoire afin de pouvoir afficher le graphque ci dessous : 
    histo = u.entrainer_(embedding, tete, X_train, Y_train, X_test, Y_test,
                                  pooling="avg", epochs=100, lr=1e-2)
    return (histo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    (voir dans le fichier annexe pour les détails)

    En PyTorch, pas de `compile` : on choisit explicitement l'optimiseur (SGD) et la
    fonction de coût (`CrossEntropyLoss`, qui applique un log-softmax en interne —
    c'est l'équivalent de `sparse_categorical_crossentropy`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Évaluation sur le jeu de test

    Question:
    - Serait-il pertinent d'effectuer le split_valid au moment de l'entraînement ? Quelle différence ?
    """)
    return


@app.cell
def _(X_test, Y_test, embedding, tete, u):
    acc = u.accuracy_(embedding, tete, X_test, Y_test, pooling="avg")
    print(f"accuracy test : {acc:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrice de confusion

    Un point important à comprendre ici est le rôle des np.argmax

    - Que se passe-t-il si on passe en encodage ***sparse*** ?
    """)
    return


@app.cell
def matrice_de_confusion(X_test, Y_test, embedding, tete, torch, u):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    noms = ['culture', 'économie', 'science', 'international', 'société', 'sport', 'politique']

    embedding.eval()
    tete.eval()
    with torch.no_grad():
        logits = u.predire_(embedding, tete, X_test, "avg")   # (N, 7)
    Y_pred = logits.argmax(dim=1).numpy()                              # (N,) classes prédites

    ConfusionMatrixDisplay.from_predictions(
        Y_test,                  # vrais labels (déjà des entiers)
        Y_pred,                  # classes prédites
        labels=range(7),
        display_labels=noms,
    )
    plt.show()
    return (plt,)


@app.cell
def _(histo, plt):

    epochs = range(1, len(histo["train"]) + 1)
    meilleure = max(range(len(histo["test"])), key=lambda i: histo["test"][i])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, histo["train"], label="train")
    plt.plot(epochs, histo["test"], label="test")
    plt.axvline(meilleure + 1, color="gray", linestyle="--",
                label=f"meilleur test : epoch {meilleure + 1} ({histo['test'][meilleure]:.3f})")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Embedding + avg pooling : train vs test")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
