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
    return nn, np, plt, torch


@app.cell
def _():
    from pathlib import Path
    import polars as pl
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Normalization
    from keras.optimizers import SGD
    from sklearn.model_selection import train_test_split

    return Dense, SGD, Sequential, keras, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Préparation des données
    """)
    return


@app.cell
def _(np, train_test_split):
    # chargement
    def load_data():
        data = []
        with open("Notebooks\corpus6.txt", encoding="utf-8") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th")
                n_que = text.count("que")
                n_sch = text.count("sch")
                n_ij = text.count("ij")
                n_gli = text.count("gli")
                n_oi = text.count("oi")
                l = len(text)
                instance = {
                    "label":label, 
                    "th" : n_th / l, 
                    "que" : n_que / l,
                    "sch" : n_sch / l,
                    "ij" : n_ij / l,
                    "gli" : n_gli / l,
                    "oi" : n_oi / l
                }
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

    # encodage des données
    X_lang = np.array(
        [[
          d['th'],
          d['que'],
          d['sch'],
          d['ij'],
          d['gli'],
          d['oi'],
         ]
        for d in data_lang]
    )

    langues = {
        'eng': 0,
        'spa': 1,
        'deu': 2,
        'nld': 3,
        'ita': 4,
        'fra': 5
    }
    y_lang = np.eye(6)[[langues[d['label']] for d in data_lang]]

    # Division du corpus 

    X_train, X_temp, y_train, y_temp = train_test_split(X_lang, y_lang, test_size=0.30, random_state=42, stratify=y_lang)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # mo.md("essayez avec et sans normalisation !")
    return X_lang, X_test, X_train, X_val, y_lang, y_test, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Réseau de neurone : Torch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLP
    """)
    return


@app.cell
def _(X_test, X_train, X_val, nn, torch, y_test, y_train, y_val):
    # X_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    # y_xor = torch.tensor([[0.], [1.], [1.], [0.]])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).argmax(dim=1).long()
    y_val_t   = torch.tensor(y_val, dtype=torch.float32).argmax(dim=1).long()
    y_test_t  = torch.tensor(y_test, dtype=torch.float32).argmax(dim=1).long()

    # MLP in pytorch (object oriented version)
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(6, 6)
            self.out = nn.Linear(6, 6)

        def forward(self, x):
            return self.out(torch.tanh(self.hidden(x)))

    mlp = MLP()

    # training in pytorch
    opt = torch.optim.SGD(mlp.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5000):
        opt.zero_grad()
        loss = loss_fn(mlp(X_test_t), y_test_t)
        loss.backward()
        opt.step()

    # pseudo-evaluation
    with torch.no_grad():
        train_acc = ((torch.argmax(mlp(X_train_t), dim=1)).float() == y_train_t).float().mean().item()
    
        val_acc = ((torch.argmax(mlp(X_val_t), dim=1)).float() == y_val_t).float().mean().item()
    
        test_acc = ((torch.argmax(mlp(X_test_t), dim=1)).float() == y_test_t).float().mean().item()
    
        hidden_train = torch.tanh(mlp.hidden(X_train_t))
        hidden_test = torch.tanh(mlp.hidden(X_test_t))
    
    print(f"Train accuracy : {train_acc:.3f}")
    print(f"Validation accuracy : {val_acc:.3f}")
    print(f"Test accuracy : {test_acc:.3f}")
    # print(f"Comparaison : Train = {hidden_train} / Test = {hidden_test}")
    return (mlp,)


@app.cell(disabled=True, hide_code=True)
def _(X_xor, hidden_xor, mlp, mo, np, plt, torch, y_xor):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
    grid = np.zeros((xx.size, 6), dtype=np.float32)
    grid[:, 0] = xx.ravel()
    grid[:, 1] = yy.ravel()

    for k in range(2, 6):
        grid[:, k] = X_xor[:, k].mean().item()

    grid = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        p = torch.softmax(mlp(grid), dim=1).argmax(dim=1).reshape(xx.shape).numpy()

    yf = y_xor.numpy()

    fig_xor, axes = plt.subplots(1, 2, figsize=(10, 4))

    for c in range(6):
        mask = yf == c
        axes[0].scatter(X_xor[mask, 0], X_xor[mask, 1], s=120, edgecolors="k", label=f"classe {c}")

    axes[0].contourf(xx, yy, p, levels=np.arange(7)-0.5, cmap="tab10", alpha=0.25)
    axes[0].set_title("Espace d'entrée : frontière de décision")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend()

    for c in range(6):
        mask = yf == c
        axes[1].scatter(hidden_xor[mask, 0], hidden_xor[mask, 1], s=120, edgecolors="k", label=f"classe {c}")

    axes[1].set_title("Espace caché")
    axes[1].set_xlabel("h1")
    axes[1].set_ylabel("h2")
    axes[1].legend()

    plt.tight_layout()
    mo.mpl.interactive(fig_xor)
    return


@app.cell(disabled=True, hide_code=True)
def _(X_xor, mo, plt, y_xor):
    from itertools import combinations

    pairs_input = list(combinations(range(6), 2))
    fig_input, axes_input = plt.subplots(3, 5, figsize=(15, 9))

    for ax_input, (i_input, j_input) in zip(axes_input.ravel(), pairs_input):
        for c_input in range(6):
            mask_input = y_xor.numpy() == c_input
            ax_input.scatter(
                X_xor[mask_input, i_input],
                X_xor[mask_input, j_input],
                s=50,
                edgecolors="k",
                label=f"{c_input}" if i_input == 0 and j_input == 1 else None
            )
        ax_input.set_xlabel(f"x{i_input+1}")
        ax_input.set_ylabel(f"x{j_input+1}")
        ax_input.set_title(f"x{i_input+1} / x{j_input+1}")

    axes_input[0, 0].legend()
    plt.tight_layout()
    mo.mpl.interactive(fig_input)
    return (combinations,)


@app.cell(disabled=True, hide_code=True)
def _(combinations, hidden_xor, mo, plt, y_xor):
    pairs_hidden = list(combinations(range(6), 2))
    fig_hidden, axes_hidden = plt.subplots(3, 5, figsize=(15, 9))

    for ax_hidden, (i_hidden, j_hidden) in zip(axes_hidden.ravel(), pairs_hidden):
        for c_hidden in range(6):
            mask_hidden = y_xor.numpy() == c_hidden
            ax_hidden.scatter(
                hidden_xor[mask_hidden, i_hidden],
                hidden_xor[mask_hidden, j_hidden],
                s=50,
                edgecolors="k",
                label=f"{c_hidden}" if i_hidden == 0 and j_hidden == 1 else None
            )
        ax_hidden.set_xlabel(f"h{i_hidden+1}")
        ax_hidden.set_ylabel(f"h{j_hidden+1}")
        ax_hidden.set_title(f"h{i_hidden+1} / h{j_hidden+1}")

    axes_hidden[0, 0].legend()
    plt.tight_layout()
    mo.mpl.interactive(fig_hidden)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neurone unique -> 6 neurones ?
    """)
    return


@app.cell
def _(X_test, X_train, nn, torch, y_test, y_train):
    def build_and_train_lang_classif(X_train, y_train, X_test, y_test):
        # building the "model"
        six_neuron = nn.Linear(6, 6)
        # activation = nn.Softmax(dim=1)
    
        # training
        opt = torch.optim.SGD(six_neuron.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss() #nn. BCEWithLogitsLoss()
    
        y_train = torch.argmax(y_train, dim=1).long()
        y_test = torch.argmax(y_test, dim=1).long()
    
        for _ in range(5000):
            opt.zero_grad()
            # probabilities = activation(six_neuron(X_train))
            loss = loss_fn(six_neuron(X_train), y_train)
            loss.backward()
            opt.step()

        six_neuron.eval()
    
        # pseudo-evaluation
        with torch.no_grad():
            acc = ((torch.argmax(six_neuron(X_test), dim=1)).float() == y_test).float().mean().item()
        return six_neuron, acc

    classif_pytorch, acc_pt = build_and_train_lang_classif(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),

        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    print(f"Accuracy test : {acc_pt:.3f}")
    print(f"Classification : {classif_pytorch}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Réseau de neurone : Keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X_lang, keras, y_lang):
    # Construction du perceptron en Keras
    model2 = Sequential([ 
        keras.layers.Input(shape=(6,)),
        Dense(24, activation='relu'),
        Dense(12, activation='relu'),
        Dense(6, activation='softmax')
    ])

    # Compilation du modèle
    model2.compile(
        optimizer=SGD(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

    # Entraînement du modèle
    model2.fit(X_lang, y_lang, 
               epochs=10, 
               verbose=1, 
               shuffle=True, 
               validation_split=0.2)

    # pseudo-Évaluation du modèle
    loss_keras2, acc_keras2 = model2.evaluate(X_lang, y_lang, verbose=0)

    print(f"Test accuracy : {acc_keras2:.3f}")
    return (model2,)


@app.cell
def _(
    Dense,
    SGD,
    Sequential,
    X_test,
    X_train,
    X_val,
    keras,
    model2,
    y_test,
    y_train,
    y_val,
):
    # Construction du perceptron en Keras
    model3 = Sequential([ 
        keras.layers.Input(shape=(6,)),
        Dense(24, activation='relu'),
        Dense(12, activation='relu'),
        Dense(6, activation='softmax')
    ])

    # Compilation du modèle
    model3.compile(
        optimizer=SGD(), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

    # Entraînement du modèle
    model3.fit(X_train, y_train, 
               epochs=10, 
               verbose=1, 
               shuffle=True,
               validation_data=(X_val, y_val))

    # pseudo-Évaluation du modèle
    loss_keras3, acc_keras3 = model2.evaluate(X_test, y_test, verbose=0)

    print(f"Test accuracy : {acc_keras3:.3f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
