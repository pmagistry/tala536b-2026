import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import utils_tp3_torch as u

    torch.manual_seed(42)
    return plt, u


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
    model, loss = u.create_model_func(30, tv.get_vocabulary(), nb_classes=len(labels))
    history = u.train_model(model, loss, X_train, Y_train,
                            epochs=400,
                            validation_data=(X_valid, Y_valid),
                            optim='adam',
                            start_from_epoch=50, patience=30)
    return history, model


@app.cell
def _(history, plt, u):
    u.plot_history(history)
    plt.show()
    return


@app.cell
def _(X_test, Y_test, model, u):
    u.evaluate_model(model, X_test, Y_test)
    return


@app.cell
def matrice_de_confusion(X_test, Y_test, labels, model, u):
    u.plot_confusion(model, X_test, Y_test, labels)
    return


@app.cell
def _(model):
    print(model)
    print("paramètres :", sum(p.numel() for p in model.parameters()))
    return


@app.cell
def _(model, tv, u):
    u.plot_embeddings(model, tv)
    return


if __name__ == "__main__":
    app.run()
