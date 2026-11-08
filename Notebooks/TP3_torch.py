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
    import utils_tp3 as u
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    return ConfusionMatrixDisplay, DataLoader, TensorDataset, nn, plt, torch, u


@app.cell
def _(DataLoader, TensorDataset, torch, u):
    corpus, labels, encodeur = u.load_data()
    tv = u.create_textvectorizer(corpus, 5000)
    train, dev, test = u.stratified_split(corpus)

    def vers_tenseurs(sous_corpus):
        X, Y = u.encode(sous_corpus, tv, encodeur)
        X = torch.as_tensor(X).cpu().long()
        Y = torch.as_tensor(Y).cpu().long()
        return X, Y

    X_train, Y_train = vers_tenseurs(train)
    X_valid, Y_valid = vers_tenseurs(dev)
    X_test, Y_test = vers_tenseurs(test)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, Y_valid), batch_size=32)
    return (
        X_test,
        X_train,
        X_valid,
        Y_test,
        Y_train,
        Y_valid,
        labels,
        train_loader,
        tv,
    )


@app.cell
def _(labels, nn, tv):
    emb = nn.Embedding(len(tv.get_vocabulary()), 200)
    dropout = nn.Dropout(0.5)
    out = nn.Linear(200, len(labels), bias=False)

    def model(x):
        h = emb(x).max(dim=1).values
        return out(dropout(h))

    parametres = list(emb.parameters()) + list(out.parameters())
    return dropout, model, parametres


@app.cell
def _(
    X_train,
    X_valid,
    Y_train,
    Y_valid,
    dropout,
    model,
    nn,
    parametres,
    torch,
    train_loader,
):
    optimizer = torch.optim.Adam(parametres)
    loss_fn = nn.CrossEntropyLoss()

    def entrainer_une_epoque():
        dropout.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    def exactitude(X, Y):
        dropout.eval()  # dropout désactivé
        with torch.no_grad(): # pas de calcul de gradients
            return (model(X).argmax(dim=1) == Y).float().mean().item()

    history = {"accuracy": [], "val_accuracy": []}

    for epoch in range(50):
        entrainer_une_epoque()
        history["accuracy"].append(exactitude(X_train, Y_train))
        history["val_accuracy"].append(exactitude(X_valid, Y_valid))
        print(epoch + 1, history["accuracy"][-1], history["val_accuracy"][-1])
    return (exactitude,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    Y_test,
    dropout,
    labels,
    model,
    plt,
    torch,
):
    dropout.eval()
    with torch.no_grad():
        pred_test = model(X_test).argmax(dim=1)

    print("exactitude test :", (pred_test == Y_test).float().mean().item())

    ConfusionMatrixDisplay.from_predictions(
        Y_test.numpy(), pred_test.numpy(),
        labels=range(len(labels)), display_labels=labels,
    )
    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    Y_test,
    dropout,
    exactitude,
    labels,
    model,
    plt,
    torch,
):
    print("exactitude test :", exactitude(X_test, Y_test))

    dropout.eval()
    with torch.no_grad():
        _pred_test = model(X_test).argmax(dim=1)

    ConfusionMatrixDisplay.from_predictions(
        Y_test.numpy(), _pred_test.numpy(),
        labels=range(len(labels)), display_labels=labels,
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
