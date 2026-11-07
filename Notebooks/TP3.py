import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    import regex
    import os
    import json
    import random
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp3 as u

    return keras, mo, plt, u


@app.cell
def _(u):
    corpus, labels, encodeur = u.load_data()
    tv = u.create_textvectorizer(corpus, 5000)
    train, dev, test = u.stratified_split(corpus)
    return dev, encodeur, labels, test, train, tv


@app.cell
def _(dev, encodeur, test, train, tv, u):
    X_train, Y_train = u.encode(train, tv, encodeur)
    X_valid, Y_valid = u.encode(dev, tv, encodeur)
    X_test, Y_test = u.encode(test, tv, encodeur)
    return X_test, X_train, X_valid, Y_test, Y_train, Y_valid


@app.cell
def _(X_train, X_valid, Y_train, Y_valid, keras, labels, tv, u):
    model = u.create_model_func(30, tv.get_vocabulary(), nb_classes=len(labels), optim='adam')
    history = model.fit(X_train, Y_train, 
                        epochs = 400,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[keras.callbacks.EarlyStopping("val_accuracy", start_from_epoch=50, patience=30)]
                       )
    return history, model


@app.cell
def _(history, plt, u):
    u.plot_history(history)
    plt.show()
    return


@app.cell
def _(X_test, Y_test, model):
    model.evaluate(X_test, Y_test, batch_size=1)
    return


@app.cell
def matrice_de_confusion(X_test, Y_test, labels, model, u):
    u.plot_confusion(model, X_test, Y_test, labels)
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(model, tv, u):
    import plotly.express as px
    from umap import UMAP

    u.plot_embeddings(model, tv, 1)
    return


@app.cell
def _(keras, mo, model):
    keras.utils.plot_model(model, show_dtype=True, show_shapes=True, show_layer_names=True, show_layer_activations=True)
    mo.image("./model.png")
    return


if __name__ == "__main__":
    app.run()
