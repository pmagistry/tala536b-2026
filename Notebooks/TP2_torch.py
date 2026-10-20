import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP2 : classification en 6 langues (version PyTorch)

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
    from sklearn.model_selection import train_test_split
    import utils_tp2_torch as u

    return mo, np, train_test_split, u


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
    dataset = u.load_data("./corpus6.txt", limit=500)
    langues = list(sorted({d['label'] for d in dataset}))
    # u.normalise_data(dataset) # on le fait à l'intérieur du réseau
    return dataset, langues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage et encodage du corpus

    Le découpage 80/20 est réalisé avec scikit-learn

    L'encodage se fait ici en ***sparse***, une version **one-hot** des fonctions et du modèle est disponible
    """)
    return


@app.cell
def _(dataset, langues, train_test_split, u):
    train, test = train_test_split(dataset, train_size=0.8, random_state=42)
    X_train, Y_train = u.encode_corpus_sparse(train, langues)
    X_test, Y_test = u.encode_corpus_sparse(test, langues)
    return X_test, X_train, Y_test, Y_train


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


@app.cell
def _(X_train, Y_train, langues, u):
    model, loss = u.model_sparse(len(langues), len(langues))
    model.normalization.adapt(X_train)  # on calcule les paramètres de normalisation sur le corpus d'entraînement
    u.train_model(model, loss, X_train, Y_train, epochs=1000, batch_size=32)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Évaluation sur le jeu de test

    Question:
    - Serait-il pertinent d'effectuer le split_valid au moment de l'entraînement ? Quelle différence ?
    """)
    return


@app.cell
def _(X_test, Y_test, model, u):
    u.evaluate_model(model, X_test, Y_test)
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
def matrice_de_confusion(X_test, Y_test, langues, model, np, u):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    Y_pred = u.predict_model(model, X_test)

    ConfusionMatrixDisplay.from_predictions(
        Y_test,  # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1),
        labels=range(len(langues)),
        display_labels=langues
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
