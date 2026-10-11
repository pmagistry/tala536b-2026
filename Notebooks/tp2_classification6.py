import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    from utilitaires import load_data, normalise_data, get_label_list, one_hot_encode

    return (
        Dense,
        SGD,
        Sequential,
        get_label_list,
        keras,
        load_data,
        mo,
        normalise_data,
        np,
        one_hot_encode,
    )


@app.cell
def _(get_label_list, load_data, mo, normalise_data, np, one_hot_encode):
    # chargement du corpus à 6 langues
    data_lang6 = load_data("./corpus6.txt")

    normalise_data(data_lang6)

    labels = get_label_list(data_lang6)

    # X : une ligne par texte, 2 colonnes (th, en)
    X_lang6 = np.array([[d['th'], d['en']] for d in data_lang6])

    y_lang6 = np.array([one_hot_encode(d['label'], labels) for d in data_lang6])

    mo.md("{} exemples chargés, {} langues : {}".format(len(data_lang6), len(labels), labels))
    return X_lang6, y_lang6


@app.cell
def _(X_lang6, mo, y_lang6):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_lang6, y_lang6, test_size=0.2, random_state=42
    )
    mo.md("{} exemples train, {} exemples test".format(len(X_train), len(X_test)))

    return X_test, X_train, y_test, y_train


@app.cell
def _(Dense, SGD, Sequential, X_test, X_train, keras, mo, y_test, y_train):
    # construction du réseau mais la sortie passe de 1 neurone à 6 neurones
    model6 = Sequential([
        keras.layers.Input(shape=(2,)),
        Dense(6, activation='softmax')
    ])

    model6.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # entraînement 
    model6.fit(X_train, y_train,
               epochs=10,
               verbose=1,
               shuffle=True)

    # évaluation finale
    loss_test, acc_test = model6.evaluate(X_test, y_test, verbose=0)
    mo.md("Précision sur le jeu de test : {:.2%}".format(acc_test))
    return


if __name__ == "__main__":
    app.run()
