import marimo

__generated_with = "0.25.1"
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
    # Du perceptron classique au deep learning


    ### Trois étapes, une même idée : apprendre une frontière de décision

    1. Le perceptron de Rosenblatt (1958)
    2. Le neurone unique entraîné par SGD (régression logistique)
    3. Le perceptron multicouche (MLP) et la non-linéarité
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Le perceptron classique (Rosenblatt, 1958)

    **Prédiction** — fonction seuil dure :

    $$
    \hat{y} = \mathbb{1}\{ w \cdot x + b > 0 \}
    $$

    **Règle de mise à jour** — exemple par exemple :

    $$
    w \leftarrow w + \eta \, (y - \hat{y}) \, x
    $$

    - Aucune correction si l'exemple est bien classé.
    - Correction $\pm x$ sinon : déplacement direct du vecteur de poids.
    - **Ni gradient, ni fonction de coût différentiable.**

    *Verrou : le signal $(y - \hat{y}) \in \{-1, 0, +1\}$ est calculé
    après un seuil dur.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Création de données
    (artificelles et linéairement séparables)

    classe 0 si

    $$ 2y + x  > 0 $$

    sinon classe 1.

    Autrement dit on sépare les données par la droite :
    $$ y = -\frac{x}{2} $$
    """)
    return


@app.cell
def _(mo, torch):

    N = 100
    X_lin = torch.randn(N, 2)
    y_lin = (X_lin[:, 0] + 2 * X_lin[:, 1] > 0).float()
    mo.md("Données artificielles, linéairement séparables dans R^2")
    return X_lin, y_lin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Entraînement d'un perceptron (méthode classique)
    """)
    return


@app.cell
def _(X_lin, mo, torch, y_lin):
    def train_perceptron(X, y, eta=1.0, max_epochs=100):
        w = torch.zeros(X.shape[1])
        b = 0.0
        for epoch in range(max_epochs):
            errors = 0
            for i in range(X.shape[0]):
                xi, yi = X[i], y[i].item()
                yhat = 1.0 if (w @ xi + b) > 0 else 0.0
                update = eta * (yi - yhat)   # dans {-1, 0, +1}
                if update != 0:
                    w += update * xi
                    b += update
                    errors += 1
            if errors == 0:
                break
        return w, b, epoch

    w_perc, b_perc, n_ep = train_perceptron(X_lin, y_lin)
    mo.md("entrainement d'un perceptron")
    return b_perc, w_perc


@app.cell
def _(np, plt):
    def plot_artifical_data(X, y, w, b, title="données et modèle linéaires"):
        fig, ax = plt.subplots(figsize=(5, 4))
        m = y == 1
        ax.scatter(X[m, 0], X[m, 1], c="tab:red", s=15, label="classe 1")
        ax.scatter(X[~m, 0], X[~m, 1], c="tab:blue", s=15, label="classe 0")
        xs = np.linspace(-3, 3, 50)
        if abs(w[1]) > 1e-8:
            ax.plot(xs, -(w[0] * xs + b) / w[1], "k-", lw=2)
        ax.set_title(title)
        ax.legend()
        plt
        return fig

    return (plot_artifical_data,)


@app.cell
def _(X_lin, b_perc, plot_artifical_data, w_perc, y_lin):
    fig_perc = plot_artifical_data(X_lin, y_lin, w_perc, b_perc, "Perceptron")
    return (fig_perc,)


@app.cell(hide_code=True)
def _(fig_perc, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## Le perceptron en action

                Données linéairement séparables : l'algorithme converge
                en un nombre fini d'erreurs.

                Mais deux limites fondamentales :
                - il **s'arrête dès la séparation parfaite** : n'importe
                  quelle frontière correcte convient, le résultat dépend
                  de l'ordre des exemples ;
                - si les données **ne sont pas séparables**, il cycle
                  indéfiniment sans jamais se stabiliser.
                """
            ),
            fig_perc,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Le neurone unique entraîné par SGD

    **Prédiction** — probabilité continue :

    $$
    p = \sigma(w \cdot x + b) \in \, ]0, 1[
    $$

    **Fonction de coût** — entropie croisée binaire, différentiable :

    $$
    L = -\left[ y \log p + (1 - y) \log(1 - p) \right]
    $$

    **Mise à jour** — descente de gradient stochastique :

    $$
    w \leftarrow w - \eta \, \frac{\partial L}{\partial w}
    $$

    Le seuil dur a disparu du calcul : le gradient
    $\partial L / \partial w = (p - y)\, x$ **transporte
    l'information d'erreur à travers tout le réseau**.

    C'est ce qui rend la méthode généralisable par
    rétropropagation — contrairement à la règle du perceptron.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Construction et entraînement du réseau en pytorch
    """)
    return


@app.cell
def _(X_lin, mo, nn, torch, y_lin):
    def build_and_train_single_neuron(X, y):
        # building the "model"
        single_neuron = nn.Linear(2, 1)

        # training
        opt = torch.optim.SGD(single_neuron.parameters(), lr=0.1)
        loss_fn = nn.BCEWithLogitsLoss()
        y_col = y.unsqueeze(1)

        for _ in range(500):
            opt.zero_grad()
            loss = loss_fn(single_neuron(X), y_col)
            loss.backward()
            opt.step()

        # pseudo-evaluation
        with torch.no_grad():
            acc = ((torch.sigmoid(single_neuron(X)) > 0.5).float() == y_col).float().mean().item()
        return single_neuron, acc
    single, acc = build_and_train_single_neuron(X_lin, y_lin)
    mo.md("code d'entraînement d'un unique neurone")
    return (single,)


@app.cell
def _(X_lin, plot_artifical_data, single, y_lin):

    fig_sgd = plot_artifical_data(X_lin, y_lin,
                                  w=single.weight.data.squeeze().numpy(),
                                  b=single.bias.data.item(), title="1 neuron, SGD")
    return (fig_sgd,)


@app.cell(hide_code=True)
def _(fig_sgd, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## SGD : même frontière linéaire, autre façon d'apprendre les poids.

                La frontière reste un hyperplan — un neurone calcule
                toujours $w \cdot x + b$. Mais :

                - la **perte est convexe** : pas de minima locaux,
                  convergence garantie (pas décroissant de Robbins-Monro) ;
                - le SGD **fonctionne sur données non séparables** :
                  il atteint le meilleur compromis linéaire au sens du
                  coût, là où le perceptron cycle ;
                - il continue à optimiser après la séparation et tend
                  vers une frontière à grande marge (lien avec les SVM).
                """
            ),
            fig_sgd,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. La limite : les données non linéairement séparables

    Exemple minimal : **XOR** — 4 points, 2 classes.

    | $x_1$ | $x_2$ | $y$ |
    |-------|-------|-----|
    | 0 | 0 | 0 |
    | 0 | 1 | 1 |
    | 1 | 0 | 1 |
    | 1 | 1 | 0 |


    Aucune droite ne sépare les classes : le neurone unique
    plafonne à **50 %** d'exactitude, quelle que soit la
    méthode d'entraînement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## La solution : une couche cachée non linéaire

    Empiler des couches **linéaires** ne sert à rien :
    la composition d'applications affines reste affine.

    $$
    \text{MLP}(x) = W_2 \, \tanh(W_1 x + b_1) + b_2
    $$

    1. $W_1$ : transformation affine de l'espace ;
    2. $\tanh$ : activation **non-linéarité** qui plie l'espace ;
    3. $W_2$ : décision linéaire **dans l'espace transformé**.

    Le réseau apprend une représentation dans laquelle
    les données redeviennent séparables.

    *Théorème d'approximation universelle (Cybenko 1989,
    Hornik 1991) : une couche cachée suffisamment large
    approxime toute fonction continue.*
    """)
    return


@app.cell
def _(nn, torch):
    # XOR Data
    X_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y_xor = torch.tensor([[0.], [1.], [1.], [0.]])

    # MLP in pytorch (object oriented version)
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 2)
            self.out = nn.Linear(2, 1)

        def forward(self, x):
            return self.out(torch.tanh(self.hidden(x)))

    mlp = MLP()

    # training in pytorch
    opt = torch.optim.SGD(mlp.parameters(), lr=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(5000):
        opt.zero_grad()
        loss = loss_fn(mlp(X_xor), y_xor)
        loss.backward()
        opt.step()

    # pseudo-evaluation
    with torch.no_grad():
        acc_mlp = ((torch.sigmoid(mlp(X_xor)) > 0.5).float() == y_xor).float().mean().item()
        hidden_xor = torch.tanh(mlp.hidden(X_xor))
    return X_xor, hidden_xor, mlp, y_xor


@app.cell(hide_code=True)
def _(X_xor, hidden_xor, mlp, np, plt, torch, y_xor):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        p = torch.sigmoid(mlp(grid)).reshape(xx.shape).numpy()

    fig_xor, axes = plt.subplots(1, 2, figsize=(10, 4))
    yf = y_xor.squeeze(1)

    axes[0].contourf(xx, yy, p, levels=20, cmap="coolwarm", alpha=0.7)
    axes[0].contour(xx, yy, p, levels=[0.5], colors="k", linewidths=2)
    axes[0].scatter(X_xor[yf == 0, 0], X_xor[yf == 0, 1], c="tab:blue", s=120, edgecolors="k", label="classe 0")
    axes[0].scatter(X_xor[yf == 1, 0], X_xor[yf == 1, 1], c="tab:red", s=120, edgecolors="k", marker="s", label="classe 1")
    axes[0].set_title("Espace d'entrée : frontière non linéaire")
    axes[0].legend()

    axes[1].scatter(hidden_xor[yf == 0, 0], hidden_xor[yf == 0, 1], c="tab:blue", s=120, edgecolors="k", label="classe 0")
    axes[1].scatter(hidden_xor[yf == 1, 0], hidden_xor[yf == 1, 1], c="tab:red", s=120, edgecolors="k", marker="s", label="classe 1")
    w = mlp.out.weight.data.squeeze().numpy()
    b = mlp.out.bias.data.item()
    hh = np.linspace(-1.2, 1.2, 50)
    axes[1].plot(hh, -(w[0] * hh + b) / w[1], "k--", lw=2)
    axes[1].set_title("Espace caché : linéairement séparable !")
    axes[1].set_xlabel("h1"); axes[1].set_ylabel("h2"); axes[1].legend()

    # plt.tight_layout()
    # mo.mpl.interactive(fig_xor)
    return (fig_xor,)


@app.cell(hide_code=True)
def _(fig_xor, mo):
    mo.vstack(
        [
            mo.md(
                f"""
                ## Démo : le MLP résout XOR 

                **À gauche** : dans l'espace d'entrée, la frontière de
                décision est courbe ou demande deux droites (impossible pour un neurone unique).

                **À droite** : les 4 points projetés dans l'espace caché $(h_1, h_2)$. 
                La couche de sortie n'a plus qu'une droite à placer : la couche cachée a rendu XOR linéairement séparable.
                """
            ),
            fig_xor,
        ]
    )
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

    # charger le corpus de 6 langues

    from pathlib import Path

    chemin = Path("corpus6.txt")
    if not chemin.exists():
        chemin = Path("../corpus6.txt")

    def load_data():
        data = []
        with open(chemin, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                label, text = line.split(" ", 1)
                longueur = max(len(text), 1)

                # les trois bigrammes 
                n_th = text.count("th")
                n_en = text.count("en")
                n_le = text.count("le")
                instance = {
                    "label": label,
                    "th": n_th / longueur,
                    "en": n_en / longueur,
                    "le": n_le / longueur,
                }
                data.append(instance)
        return data

    data_lang = load_data()


    # Comme on 6 langues on attrbue un numéro à chaque langue

    lang_to_id = {
        "deu": 0,  
        "eng": 1,  
        "fra": 2, 
        "ita": 3,  
        "spa": 4,  
        "nld": 5, 
    }

  
    X_lang = np.array(
        [[d["th"], d["en"], d["le"]] for d in data_lang],
        dtype="float32",
    )
    y_lang = np.array(
        [lang_to_id[d["label"]] for d in data_lang],
        dtype="int64",
    )

    mo.md(f"**TP2 :** {len(data_lang)} textes, 6 langues, 3 bigrammes.")
    return X_lang, y_lang


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Séparation des données et classification Keras en six langues
    """)
    return


@app.cell
def _(X_lang, mo, np, y_lang):

    # séparer les données AVANT de les normaliser

    from sklearn.model_selection import train_test_split

    # 80 % pour apprendre, 20 % réservés au test final.
    X_train, X_test, y_train, y_test = train_test_split(
        X_lang, y_lang,
        test_size=0.20,
        random_state=42,
        stratify=y_lang,  
    )

   
    #  normalisation 
    moyenne = np.mean(X_train, axis=0)
    ecart_type = np.std(X_train, axis=0)

    ecart_type[ecart_type == 0] = 1

    X_train = (X_train - moyenne) / ecart_type
    X_test = (X_test - moyenne) / ecart_type

    mo.md(
        f"**Séparation :** {len(X_train)} textes pour l'entraînement "
        f"et {len(X_test)} textes pour le test."
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en Keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X_test, X_train, keras, mo, y_test, y_train):

    # réseau Keras pour 6 langues
    model = Sequential([
        keras.layers.Input(shape=(3,)),   # 3 fréquences : th, en, le
        Dense(8, activation="tanh"),      # couche cachée 
        Dense(6, activation="softmax"),   # softmax quand on a plus de 2 langues
    ])


    # Dans cet exemple on utilise sparse_categorical_crossentropy à la place de BCE
    model.compile(
        optimizer=SGD(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train, y_train,
        epochs=10,
        verbose=1,
        shuffle=True,
        validation_split=0.2,
    )

   
    loss_keras, acc_keras = model.evaluate(X_test, y_test, verbose=0)

    mo.md(
        f"**TP2 — précision sur les textes de test : {acc_keras:.2%}**"
    )
    return


if __name__ == "__main__":
    app.run()
