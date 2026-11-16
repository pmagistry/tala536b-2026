import marimo

__generated_with = "0.25.0"
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
    import utils_tp4 as u

    return mo, plt, u


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
def _(X_train, X_valid, Y_train, Y_valid, labels, tv, u):
    model = u.create_model_func(30, tv.get_vocabulary(), nb_classes=len(labels), optim='adam')

    train_loader = u.create_dataloader(
        X_train, Y_train, batch_size=32, shuffle=True
    )

    valid_loader = u.create_dataloader(
        X_valid, Y_valid, batch_size=32, shuffle=False
    )

    # Entraînement
    history = u.train_model(
        model,
        train_loader,
        valid_loader,
        epochs=400,
        optim='adam',
        lr=0.001
    )
    return history, model


@app.cell
def _(history, plt, u):
    u.plot_history(history)
    plt.show()
    return


@app.cell
def _(X_test, Y_test, model, u):
    u.evaluate_model(model, X_test, Y_test, batch_size=1)
    return


@app.cell
def matrice_de_confusion(X_test, Y_test, labels, model, u):
    u.plot_confusion(model, X_test, Y_test, labels)
    return


@app.cell
def _(model):
    from torchinfo import summary

    summary(model, input_size=(1, 30))
    return


@app.cell
def _(model, tv, u):
    import plotly.express as px
    from umap import UMAP

    u.plot_embeddings(model, tv, 0)
    return


@app.cell
def _(mo, model):
    from torchview import draw_graph
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    model_graph = draw_graph(
        model,
        input_size=(1, 30),
        dtypes=[torch.long],
        graph_name="DeepTextClassifier",
        device=DEVICE,
        expand_nested=True
    )

    model_graph.visual_graph.render(
        filename="model",
        format="png",
        cleanup=True
    )

    mo.image("./model.png")
    return


if __name__ == "__main__":
    app.run()
