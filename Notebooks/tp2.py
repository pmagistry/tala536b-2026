import marimo

__generated_with = "0.25.1"
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
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # torch.manual_seed(0)
    return ConfusionMatrixDisplay, confusion_matrix, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Données réelles et implémentation en Keras (et en pytorch)
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
    # chargement
    def load_data():
        data = []
        with open("../../data/corpus6.txt") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th")
                n_en = text.count("en")
                n_le = text.count("le")
                l = len(text)
                instance = {"label":label, "th": n_th / l, "en": n_en /l, "le": n_le}
                data.append(instance)
        return data

    nb_lines = 2000
    eval = nb_lines//10
    data_lang = load_data()[:nb_lines]

    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k]- mean) / std

    normalise_data(data_lang)
    return data_lang, eval


@app.cell
def _(data_lang, np, torch):
    # encodage des données
    X_lang = np.array([[d['th'], d['en'], d['le']] for d in data_lang])

    langs = ["deu", "nld", "fra", "eng", "spa", "ita"]
    lang_to_idx = {lang: i for i, lang in enumerate(langs)}
    langs_matrix = np.eye(6)

    # j'ai trouvé ça assez intuitif de générer le gold ainsi (pour torch beaucoup moins)
    y_lang_test = np.array([langs_matrix[lang_to_idx[d["label"]]] for d in data_lang])

    X_lang2 = torch.tensor([[d['th'], d['en'], d['le']] for d in data_lang])

    langs_matrix2 = torch.eye(6)

    # j'ai utilisé l'ia pour comprendre comment faire la même chose que j'ai fait en numpy. J'ai eu du mal à comprendre comment cette solution fonctionnait, notamment à cause de la boucle de 500 itérations que torch fait en interne pour construire le tenseur final. De ce que j'ai compris c'est plus rapide de faire ainsi
    y_lang2 = langs_matrix2[[lang_to_idx[d["label"]] for d in data_lang]]

    # visiblement on peut faire la même chose avec numpy, c'est ce que j'ai utilisé au final
    y_lang = langs_matrix[[lang_to_idx[d["label"]] for d in data_lang]]
    return X_lang, X_lang2, langs, y_lang, y_lang2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en pytorch
    """)
    return


@app.cell
def _(X_lang2, nn, torch, y_lang2):
    # je n'ai pas encore réussi à faire fonctionner avec pytorch, c'est en cours
    def build_and_train_lang_classif(X, y):
        # building the "model"
        model = nn.Sequential(
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Softmax()
        )
        # training
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MultiLabelMarginLoss()
        y_col = y.unsqueeze(1)

        for _ in range(10):
            opt.zero_grad()
            loss = loss_fn((model(X)), y)
            loss.backward()
            opt.step()

        # pseudo-evaluation
        with torch.no_grad():
            acc = ((torch.sigmoid(model(X)) > 0.5).float() == y_col).float().mean().item()
        return model, acc
    classif_pytorch, acc_pt = build_and_train_lang_classif(torch.tensor(X_lang2, dtype=torch.float32), torch.tensor(y_lang2, dtype=torch.float32))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en Keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X_lang, eval, keras, mo, y_lang):
    # Construction du perceptron en Keras
    model = Sequential([
        keras.layers.Input(shape=(3,)),
        Dense(12, activation='tanh'),  # j'ai opté pour 12 neurones parallèles, un peu au hasard honnêtement 
        Dense(6, activation='softmax')
    ])

    # Compilation du modèle
    model.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model.fit(X_lang[:-eval], y_lang[:-eval],
                        epochs=20, 
                        verbose=0, 
                        shuffle=True, 
                        validation_split=0.2)

    # Évaluation du modèle (avec 10% du corpus qui y est alloué)
    loss_keras, acc_keras = model.evaluate(X_lang[-eval:], y_lang[-eval:], verbose=0)

    # Affichage des poids appris
    weights = model.layers[0].get_weights()[0]
    bias = model.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras:.2%}")
    return (model,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_lang,
    confusion_matrix,
    eval,
    langs,
    model,
    np,
    plt,
    y_lang,
):
    y_true = [langs[i] for i in np.argmax(y_lang[-eval:], axis=1)]
    y_pred = [langs[i] for i in np.argmax(model.predict(X_lang[-eval:], verbose=0), axis=1)]

    cm = confusion_matrix(y_true, y_pred, labels=langs)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=langs)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
