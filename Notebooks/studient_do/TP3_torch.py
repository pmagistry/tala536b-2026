import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    from torch.utils.data import TensorDataset, DataLoader
    import matplotlib.pyplot as plt

    import regex
    import os
    import json
    import random
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp3_torch as u

    return DataLoader, TensorDataset, u


@app.cell
def _(u):
    corpus, labels, encodeur = u.load_data(corpus_file="corpus/corpus-classif.json")
    tv = u.create_textvectorizer(corpus, 5000)
    vocab = tv.get_vocabulary()
    train, dev, test = u.stratified_split(corpus)
    return dev, encodeur, test, train, tv, vocab


@app.cell
def _(dev, encodeur, test, train, tv, u):
    X_train, Y_train = u.encode(train, tv, encodeur)
    X_valid, Y_valid = u.encode(dev, tv, encodeur)
    X_test, Y_test = u.encode(test, tv, encodeur)
    return X_test, X_train, X_valid, Y_test, Y_train, Y_valid


@app.cell
def _(DataLoader, TensorDataset, X_train, X_valid, Y_train, Y_valid):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    dev_loader = DataLoader(TensorDataset(X_valid, Y_valid), batch_size=32)
    return dev_loader, train_loader


@app.cell
def _(encodeur, u, vocab):
    model = u.ModelFunc(vocab_size=len(vocab), emb_size=200, nb_classes=len(encodeur), dropout=0.5)
    return (model,)


@app.cell
def _(model, u, vocab):
    model.embedding = u.initialize_embeddings(model.embedding, vocab, "studient_do/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin")
    return


@app.cell
def _(dev_loader, model, train_loader, u):
    train_model, history = u.train_model(model, train_loader, dev_loader, epochs=500)
    return (history,)


@app.cell
def _(history, u):
    u.plot_history(history)
    return


@app.cell
def _(X_test, Y_test, encodeur, model, u):
    u.plot_confusion(model, X_test, Y_test, encodeur)
    return


@app.cell
def _(model, u, vocab):
    u.plot_embeddings(model, vocab)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
