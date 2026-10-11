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
    return mo, nn, np, plt, torch


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
def _(mo, np):

    # chargement
    def load_data():
        data = []
        with open("./tala536b-2026/Notebooks/corpus6.txt") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th") # Anglais
                n_del = text.count("del") # Espagnol
                n_der = text.count("der") # Allemand
                n_ij = text.count("ij") # Néerlandais
                n_gli = text.count("gli") # Italien
                n_le = text.count("le") # Français
                l = len(text)
                instance = {
                    "label":label, 
                    "th": n_th / l, 
                    "del": n_del / l,
                    "der": n_der / l, 
                    "ij": n_ij / l,
                    "gli": n_gli / l,
                    "le": n_le / l
                }
                data.append(instance)
        return data

    data_lang = load_data()[:2000]

    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k]- mean) / std

    normalise_data(data_lang)

    # encodage des données
    X_lang = np.array([
        [
            d['th'], 
            d['del'],
            d['der'],
            d['ij'],
            d['gli'],
            d['le'],
        ] 
         for d in data_lang
                      ])

    langues = {
        "eng" : 0,
        "spa" : 1,
        "deu" : 2,
        "nld" : 3,
        "ita" : 4,
        "fra" : 5
    }

    y_lang = np.eye(6)[[langues[d['label']] for d in data_lang]]

    mo.md("essayez avec et sans normalisation !")
    return X_lang, y_lang


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mise en place de l'évaluation par sklearn
    """)
    return


@app.cell
def _():
    import sklearn.model_selection as sk_mod
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    return ConfusionMatrixDisplay, confusion_matrix, sk_mod


@app.cell
def _(X_lang, sk_mod, y_lang):
    X_train, X_temp, y_train, y_temp = sk_mod.train_test_split(X_lang, y_lang)
    X_dev, X_test, y_dev, y_test = sk_mod.train_test_split(X_temp, y_temp)
    return X_dev, X_test, X_train, y_dev, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction du modèle avec **Pytorch**
    """)
    return


@app.cell
def _(X_test, X_train, nn, torch, y_test, y_train):
    def build_and_train_lang_classif(X_train, y_train, X_test, y_test):
        # building the "model"
        poly_neuron = nn.Sequential(
            nn.Linear(6, 8),
            nn.Tanh(),
            nn.Linear(8,6)
        )

        # training
        opt = torch.optim.SGD(poly_neuron.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        y_col = torch.argmax(y_train, dim=1)

        for _ in range(1000):
            opt.zero_grad()
            logits = poly_neuron(X_train)               
            loss = loss_fn(logits, y_col)
            loss.backward()
            opt.step()

        # pseudo-evaluation
        with torch.no_grad():
            logits = poly_neuron(X_test)
            predictions = torch.argmax(logits, dim=1)
            acc = (predictions == torch.argmax(y_test, dim=1)).float().mean().item()
        return poly_neuron, acc

    classif_pytorch, acc_pt = build_and_train_lang_classif(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32)   
    )
    return (acc_pt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLP en **keras**
    """)
    return


@app.cell
def _(
    Dense,
    SGD,
    Sequential,
    X_dev,
    X_test,
    X_train,
    acc_pt,
    keras,
    mo,
    y_dev,
    y_test,
    y_train,
):
    # Construction du perceptron en Keras
    model2 = Sequential([
        keras.layers.Input(shape=(6,)),
        Dense(8, activation='tanh'),
        Dense(6, activation='softmax')
    ])

    # Compilation du modèle
    model2.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model2.fit(
        X_train, y_train,
        epochs=1000, 
        verbose=1, 
        shuffle=True, 
        validation_data=(X_dev, y_dev)
    )

    # pseudo-Évaluation du modèle
    loss_keras2, acc_keras2 = model2.evaluate(X_test, y_test, verbose=0)

    # Affichage des poids appris
    weights2 = model2.layers[0].get_weights()[0]
    bias2 = model2.layers[0].get_weights()[1]
    mo.md(f"Précision avec PyTorch : {acc_pt:.2%}\n\n"
            f"Précision avec Keras : {acc_keras2:.2%}\n\n"
            f"Poids pour 'th' : {weights2[0]}\n\n"
            f"Poids pour 'del' : {weights2[1]}\n\n"
            f"Poids pour 'der' : {weights2[2]}\n\n"
            f"Poids pour 'ij' : {weights2[3]}\n\n"
            f"Poids pour 'gli' : {weights2[4]}\n\n"
            f"Poids pour 'le' : {weights2[5]}\n\n"
            f"Biais : {bias2}")
    return acc_keras2, model2


@app.cell
def _(acc_keras2, acc_pt, mo):
    def call_(acc_keras2, acc_pt, mo):
        affichage = mo.md(
            f"### Comparaison des modèles MLP\n\n"
            f"- Précision avec **PyTorch** : **{acc_pt:.2%}**\n"
            f"- Précision avec **Keras** : **{acc_keras2:.2%}**"
        )
        return affichage

    call_(acc_keras2, acc_pt, mo)
    return


@app.cell
def _(acc_keras2, acc_pt):
    (acc_keras2, acc_pt)
    return


@app.cell
def _(model2):
    model2.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrice de confusion
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix):
    def matrice_condusion(model2, X_test, y_test, np, plt):
        prediction_prob = model2.predict(X_test)
        y_prob = np.argmax(prediction_prob, axis=1)
        y_true = np.argmax(y_test, axis = 1)

        noms_langues = ["eng", "spa", "deu", "nld", "ita", "fra"]
        matrice = confusion_matrix(y_prob, y_true)

        fig, ax = plt.subplots(figsize=(8,6))
        titre = "Matrice de Condusion de Keras"
        affichage = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=noms_langues)
        affichage.plot(ax=ax, cmap='Blues')

        ax.set_title(titre)

        return fig

    return (matrice_condusion,)


@app.cell
def _(X_test, matrice_condusion, model2, np, plt, y_test):
    matrice_condusion(model2, X_test, y_test, np, plt)
    return


if __name__ == "__main__":
    app.run()
