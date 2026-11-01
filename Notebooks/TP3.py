import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    # torch.manual_seed(0)
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports pour keras
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path
    import polars as pl
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Input, TextVectorization, Embedding, GlobalAveragePooling1D, GlobalMaxPool1D
    from keras.optimizers import SGD

    mo.md("une variable d'environnement vient définir le backend qui sera utilisé par Kera (torch, tensorflow ou jax)")
    return (keras,)


@app.cell
def _():
    import utils_tp2 as u_ke
    import utils_tp2_torch as u_py

    return u_ke, u_py


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chargement de données (disponibles sur Moodle)
    """)
    return


@app.cell
def _(u_ke):
    corpus = u_ke.load_text("./Notebooks/corpus6.txt", limit=500)
    langues = list(sorted({ d['label'] for d in corpus}))
    code_langues = {l:i for i, l in enumerate(langues)}
    return code_langues, corpus, langues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mise en place de l'évaluation par sklearn
    """)
    return


@app.cell
def _():
    import sklearn.model_selection as sk_mod
    from sklearn.metrics import ConfusionMatrixDisplay

    return ConfusionMatrixDisplay, sk_mod


@app.cell
def _(corpus, keras, sk_mod):
    train_ke, test_ke = sk_mod.train_test_split(corpus, train_size=8, random_state=42)
    tv = keras.layers.TextVectorization(
        max_tokens=10,
        standardize=None,
        split="character",
        output_mode="int",
        output_sequence_lenth=100
    )
    tv.adapt([line['text'] for line in train_ke])
    return test_ke, train_ke, tv


@app.cell
def _(code_langues, np, test_ke, train_ke, tv):
    X_train_ke = tv([line['text'] for line in train_ke])
    Y_train_ke = np.array([code_langues[line['label']] for line in train_ke])

    X_test_ke = tv([line['text'] for line in test_ke])
    Y_test_ke = np.array([code_langues[line['label']] for line in test_ke])
    return X_test_ke, X_train_ke, Y_test_ke, Y_train_ke


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction du modèle avec **Pytorch**
    """)
    return


@app.cell
def _(sk_mod, u_py):
    dataset = u_py.load_data("./Notebooks/corpus6.txt", limit=500)
    langues_py = list(sorted({ d['label'] for d in dataset}))

    train_py, test_py = sk_mod.train_test_split(dataset, train_size=0.8, random_state=42)
    X_train_py, Y_train_py = u_py.encode_corpus_sparse(train_py, langues_py)
    X_test_py, Y_test_py = u_py.encode_corpus_sparse(test_py, langues_py)

    model_py, loss_py = u_py.model_sparse(len(langues_py), len(langues_py))
    model_py.normalization.adapt(X_train_py)
    u_py.train_model(model_py, loss_py, X_train_py, Y_train_py, epochs=1000, batch_size=32)
    return X_test_py, Y_test_py, langues_py, model_py


@app.cell
def _(X_test_py, Y_test_py, model_py, u_py):
    u_py.evaluate_model(model_py, X_test_py, Y_test_py)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLP en **keras**
    """)
    return


@app.cell
def _(X_train_ke, Y_train_ke, langues, tv, u_ke):
    # Construction du perceptron en Keras

    model_ke = u_ke.model_sparse(tv.vocabulary_size(), len(langues))
    model_ke.layers[0].adapt(X_train_ke)
    model_ke.fit(X_train_ke, Y_train_ke, epochs=100, batch_size=32)
    return (model_ke,)


@app.cell
def _(X_test_ke, Y_test_ke, model_ke):
    model_ke.evaluate(X_test_ke, Y_test_ke)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrice de confusion pour Keras
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test_ke,
    Y_test_ke,
    langues,
    model_ke,
    np,
    plt,
):
    Y_pred = model_ke.predict(X_test_ke)
    ConfusionMatrixDisplay.from_predictions(Y_test_ke, np.argmax(Y_pred, axis=1), labels=range(len(langues)), display_labels=langues)
    plt.show()
    return


@app.cell
def _(model_ke):
    model_ke.summary()
    return


@app.cell
def _(tv):
    tv.get_vocabulary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrice de confusion pour **Pytorch**
    """)
    return


@app.function
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


@app.cell
def _(X_test_py, Y_test_py, langues_py, model_py, np, u_py):
    matrice_de_confusion(X_test_py, Y_test_py, langues_py, model_py, np, u_py)
    return


if __name__ == "__main__":
    app.run()
