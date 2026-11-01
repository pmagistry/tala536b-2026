import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP3 : classification des articles de presse

    ## Imports usuels
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
    return keras, mo, np, train_test_split, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lecture des données
    """)
    return


@app.cell
def _(u):
    label = "catégorie"
    corpus = u.load_text("./corpus-classif.json", label=label)
    langues = list(sorted({d["label"] for d in corpus}))
    code_langues = {l: i for i, l in enumerate(langues)}
    return code_langues, corpus, langues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage et encodage du corpus

    Le découpage 80/20 est réalisé avec scikit-learn.
    """)
    return


@app.cell
def _(corpus, keras, train_test_split):
    train, test = train_test_split(corpus, train_size=0.8, random_state=42, stratify=[d["label"] for d in corpus])
    tv = keras.layers.TextVectorization(
        max_tokens=5000,
        standardize=None,
        split="whitespace",
        output_mode="count"
    )
    tv.adapt([line["text"] for line in train])
    return test, train, tv


@app.cell
def _(code_langues, np, test, train, tv):
    X_train = tv([line["text"] for line in train])
    Y_train = np.array([code_langues[line["label"]] for line in train])
    X_test = tv([line["text"] for line in test])
    Y_test = np.array([code_langues[line["label"]] for line in test])
    return X_test, X_train, Y_test, Y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    """)
    return


@app.cell
def _(X_train, Y_train, langues, tv, u):
    model = u.model_sparse(tv.vocabulary_size(), len(langues))
    model.layers[0].adapt(X_train)
    model.fit(X_train, Y_train, epochs=100, batch_size=32)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Évaluation sur le jeu de test
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
    """)
    return


@app.cell
def matrice_de_confusion(X_test, Y_test, langues, model, np):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    Y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        Y_test, np.argmax(Y_pred, axis=1),
        labels=range(len(langues)), display_labels=langues
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nouvelle vectorisation : séquences d'identifiants et Embedding
    """)
    return


@app.cell
def _(keras, test, train):
    tv_int = keras.layers.TextVectorization(
        max_tokens=5000,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=200
    )
    tv_int.adapt([line["text"] for line in train])
    X_train_int = tv_int([line["text"] for line in train])
    X_test_int = tv_int([line["text"] for line in test])
    return X_test_int, X_train_int, tv_int


@app.cell
def _(X_train_int, Y_train, langues, tv_int, u):
    model_embedding = u.model_embedding(
        200, len(langues), tv_int.vocabulary_size(), pooling="max"
    )
    model_embedding.fit(X_train_int, Y_train, epochs=20, batch_size=32)
    return (model_embedding,)


@app.cell
def _(X_test_int, Y_test, model_embedding):
    model_embedding.evaluate(X_test_int, Y_test)
    return


@app.cell
def _(X_test_int, Y_test, langues, model_embedding, np):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    Y_pred_embedding = model_embedding.predict(X_test_int)
    ConfusionMatrixDisplay.from_predictions(
        Y_test, np.argmax(Y_pred_embedding, axis=1),
        labels=range(len(langues)), display_labels=langues
    )
    plt.show()
    return


@app.cell
def _(model, model_embedding, tv, tv_int):
    model.summary()
    model_embedding.summary()
    tv.get_vocabulary()
    tv_int.get_vocabulary()
    return


if __name__ == "__main__":
    app.run()
