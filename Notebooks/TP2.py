import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP : Ennoncé 20 octobre


    ## Imports usuels

    (ici j'utilise un fichier python annexe pour une partie de mon code, pensez à aller le voir aussi)
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp2 as u
    import json 

    return json, keras, mo, np, train_test_split, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lecture des données

    (fonctions issue du TP précédent, déplacées dans la bibliothèque)
    """)
    return


@app.cell
def _(json):

    with open("Notebooks/Corpus/corpus-classif.json", encoding="utf-8") as fichier_json:
        documents_json = json.load(fichier_json)

    corpus = []
    for doc_json in documents_json:
        texte_doc = doc_json["texte"].strip()
        rubrique_doc = doc_json["catégorie"]
        if texte_doc == "":
            continue
        if rubrique_doc == "":
            continue
        instance = {"label": rubrique_doc, "text": texte_doc}
        corpus.append(instance)

    #corpus = u.load_text("./corpus6.txt", limit=500)
    langues = list(sorted({ d['label'] for d in corpus }))
    code_langues = {l:i for i, l in enumerate(langues)}
    # u.normalise_data(dataset) # on le fait à l'intérieur du réseau

    print(len(corpus))
    return code_langues, corpus, langues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage et encodage du corpus

    Le découpage 80/20 est réalisé avec scikit-learn

    L'encodage se fait ici en ***one-hot***, une version **sparse** des fonctions et du modèle sont disponibles
    """)
    return


@app.cell
def _(corpus, keras, train_test_split):
    train, test = train_test_split(corpus, train_size=0.8, random_state=42)
    train, valid = train_test_split(train, train_size=0.8, random_state=42)
    tv = keras.layers.TextVectorization(
            max_tokens=5000, #fait 10, 100 10000 , 5000 pour surapprentissage observation
            standardize=None,
            split="whitespace",
            ngrams=None,
            output_mode="count",
        )
    tv.adapt([line['text'] for line in train])
    return test, train, tv, valid


@app.cell
def _(code_langues, np, test, train, tv, valid):

    X_train = tv([line['text'] for line in train])
    Y_train = np.array([code_langues[line['label']] for line in train])

    X_test = tv([line['text'] for line in test])
    Y_test = np.array([code_langues[line['label']] for line in test])

    X_valid = tv([line['text'] for line in valid])
    Y_valid = np.array([code_langues[line['label']] for line in valid])
    return X_test, X_train, X_valid, Y_test, Y_train, Y_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    (voir dans le fichier annexe pour les détails)
    """)
    return


@app.cell
def _(X_train, X_valid, Y_train, Y_valid, langues, tv, u):
    model = u.model_sparse(tv.vocabulary_size(),len(langues))
    model.layers[0].adapt(X_train)
    historique = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_valid, Y_valid))
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
def _(X_test, Y_test, model):
    model.evaluate(X_test, Y_test)
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
def matrice_de_confusion(X_test, Y_test, langues, model, np):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    Y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        Y_test, # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1), 
        labels=range(len(langues)), 
        display_labels=langues
    )
    plt.show()
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(tv):
    tv.get_vocabulary()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Partie avec dropout
    """)
    return


@app.cell
def _(X_train, X_valid, Y_train, Y_valid, langues, tv, u):
    model_drop = u.model_sparse_dropout(tv.vocabulary_size(), len(langues), rate=0.2)
    model_drop.layers[0].adapt(X_train)
    historique_drop = model_drop.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_valid, Y_valid))
    return (model_drop,)


@app.cell
def _(X_test, Y_test, model_drop):
    model_drop.evaluate(X_test, Y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Partit tentative embeding
    """)
    return


@app.cell
def _(keras, test, train, valid):
    tv_int = keras.layers.TextVectorization(
            max_tokens=5000,
            standardize=None,
            split="whitespace",
            ngrams=None,
            output_mode="int",
            output_sequence_length=100,
        )
    tv_int.adapt([line['text'] for line in train])

    X_train_int = tv_int([line['text'] for line in train])
    X_valid_int = tv_int([line['text'] for line in valid])
    X_test_int = tv_int([line['text'] for line in test])

    print(X_train_int.shape, X_valid_int.shape, X_test_int.shape)
    return X_test_int, X_train_int, X_valid_int, tv_int


@app.cell
def _(X_train_int, X_valid_int, Y_train, Y_valid, keras, langues, tv_int, u):
    model_emb = u.model_embedding(
        tv_int.vocabulary_size(), len(langues),
        embedding_dim=32, pooling="max",
        optim=keras.optimizers.SGD(learning_rate=1.0),
    )
    historique_emb = model_emb.fit(X_train_int, Y_train, epochs=100, batch_size=32, validation_data=(X_valid_int, Y_valid))
    return (model_emb,)


@app.cell
def _(X_test_int, Y_test, model_emb):
    model_emb.evaluate(X_test_int, Y_test)
    return


if __name__ == "__main__":
    app.run()
