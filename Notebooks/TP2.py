import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    # torch.manual_seed(0)
    return (np,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Données réelles et implémentation en Keras
    """)
    return


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
    from keras.layers import Dense, Normalization
    from keras.optimizers import SGD

    mo.md("une variable d'environnement vient définir le backend qui sera utilisé par Kera (torch, tensorflow ou jax)")
    return Dense, SGD, Sequential, keras


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chargement de données (disponibles sur Moodle)
    """)
    return


@app.cell
def _(np):
    #adaptation 6 langues
    # chargement
    def load_data():
        data = []
        with open("./corpus6.txt") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th")
                n_ou = text.count("ou")
                n_ij = text.count("ij")
                n_ch = text.count("ch")
                n_io = text.count("ió")
                n_zi = text.count("zi")
                l = len(text)
                instance = {"label":label, "th": n_th / l, "ou": n_ou / l, "ij": n_ij / l,
                            "ch": n_ch / l, "ió": n_io / l, "zi": n_zi / l}
                data.append(instance)
        return data

    data_lang = load_data()[:200]

    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k]- mean) / std

    normalise_data(data_lang)
    label2id = {"fra": 0, "eng": 1, "nld": 2, "deu": 3, "spa": 4, "ita": 5}
    # encodage des données
    X_lang = np.array([[d['th'], d['ou'], d['ij'], d['ch'], d['ió'], d['zi']] for d in data_lang])
    y_lang = np.array([label2id[d['label']] for d in data_lang])
    return X_lang, y_lang


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en Keras - 6 langues multicouche
    """)
    return


@app.cell(disabled=True)
def _(Dense, SGD, Sequential, X_lang, keras, mo, y_lang):
    model2 = Sequential([
        keras.layers.Input(shape = (6, )),
        Dense(8, activation = 'tanh'),
        Dense(6, activation = 'softmax')
    ])

    # Compilation du modèle
    model2.compile(
        optimizer = SGD(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    model2.fit(X_lang, y_lang,
                        epochs = 50, 
                        verbose = 1, 
                        shuffle = True, 
                        validation_split = 0.2)

    # pseudo-Évaluation du modèle
    loss_keras2, acc_keras2 = model2.evaluate(X_lang, y_lang, verbose = 0)

    mo.md(f"Précision du modèle : {acc_keras2:.2%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expliquer ci-dessous vos difficultés ou posez vos questions

    1.j'ai trouvé deux fonctions de loss à combiner avec 'softmax' : sparse_categorical_crossentropy et categorical_crossentropy, je n'ai pas bien compris la différence. J'ai fait un choix au hasard.                                           2. je n'ai aucune idée de comment réaliser une évaluation.
    Sinon, dans l'ensemble je comprends bien les cours et la partie théorique, ainsiq ue les maths, et en lisant les références que vous aviez données, j'essaie de combler mes lacunes(je n'ai pas fait le M1). Cependant au moment de faire les TPs, je me sens assez perdu avec le code...je ne sais même pas exactement ce que je ne comprends pas, mais tout devient assez nébuleux, je finis par bidouiller du code par imitation. J'espère qu'au fur et à mesure cela changera.
    -
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
