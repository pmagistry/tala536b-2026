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
    from sklearn.model_selection import train_test_split

    return mo, nn, np, plt, torch, train_test_split


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
def _(mo, np, train_test_split):

    bigrams = ["th", "en", "wh",   # anglais
               "ch", "ei", "sc",   # allemand
               "ij", "aa", "oe",   # néerlandais
               "ou", "eu", "qu", "oi", "le", "de", "on", "ai", "me", "it",   # français
               "zi", "gl", "zz",   # italien
               "ll", "ue", "ci"]   # espagnol
    #bigrams = ["th", "en"]
    # chargement
    def load_data():
        data = []
        with open("./corpus6.txt") as file:
            for line in file:
                line = line.strip()
                if " " not in line:   # 空行，或者只有标签没有文本
                    continue
                label, text = line.split(" ", 1)
                if len(text) == 0:
                    continue
                l = len(text)
                instance = {"label":label}
                for bg in bigrams:
                    instance[bg] = text.count(bg) / l
                data.append(instance)
        return data

    from collections import Counter

    data_test_labels = load_data()
    print(Counter(d["label"] for d in data_test_labels))

    data_lang = load_data()
    print(len(data_lang))

    # normalisation
    def normalise_data(train, test):
        for k in train[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in train])
                std = np.std([d[k] for d in train]) or 1.0 # éviter l'occasion où quelques bigrammes n'apparaîssent pas
                for d in train + test:
                    d[k] = (d[k]- mean) / std

    data_train, data_test = train_test_split(
        data_lang, test_size=0.2, random_state=42,
        stratify=[d['label'] for d in data_lang]) # test 20%, même proportion pour toutes les langues

    #normalise_data(data_lang)

    # encodage des données
    langs = sorted({d['label'] for d in data_lang})   # ['deu','eng','fra','ita','nld','spa']
    X_train = np.array([[d[bg] for bg in bigrams] for d in data_train])
    y_train = np.array([langs.index(d['label']) for d in data_train])
    X_test  = np.array([[d[bg] for bg in bigrams] for d in data_test])
    y_test  = np.array([langs.index(d['label']) for d in data_test])
    mo.md("essayez avec et sans normalisation !")
    return X_test, X_train, langs, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en Keras
    """)
    return


@app.cell
def _(
    Dense,
    SGD,
    Sequential,
    X_test,
    X_train,
    keras,
    langs,
    mo,
    y_test,
    y_train,
):
    # Construction du perceptron en Keras
    model = Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        Dense(32, activation='tanh'),
        Dense(len(langs), activation='softmax')
    ])

    # Compilation du modèle
    model.compile(
        optimizer=SGD(learning_rate=0.1),
        loss='sparse_categorical_crossentropy',# 原来binary_crossentropy
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model.fit(X_train, y_train,
                        epochs=50, 
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2)

    # pseudo-Évaluation du modèle
    loss_keras, acc_keras = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    # Affichage des poids appris
    #weights = model.layers[0].get_weights()[0]
    #bias = model.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras:.2%}")
    return (y_pred,)


@app.cell
def _(langs, y_pred, y_test):
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=langs,
        normalize="true", values_format=".2f", cmap="Blues").figure_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercices
    ## classification de langues, version pytorch
    """)
    return


@app.cell
def _(X_test, X_train, langs, nn, torch, y_test, y_train):
    def build_and_train_lang_classif(X, y, n_classes):
        # building the "model"
        #single_neuron = nn.Linear(3, 1)
        #activation = nn.Sigmoid()
        mlp6 = nn.Sequential(
                nn.Linear(X.shape[1], 32),
                nn.Tanh(),
                nn.Linear(32, n_classes),   # pas de softmax : CrossEntropyLoss l'inclut
            )
        # training
        opt = torch.optim.SGD(mlp6.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss， 二元交叉熵损失。

        for _ in range(500):
            opt.zero_grad()
            loss = loss_fn(mlp6(X),y)
            loss.backward()
            opt.step()
        return mlp6

    classif_pytorch = build_and_train_lang_classif(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            len(langs))

        # pseudo-evaluation
    with torch.no_grad():
            pred_pt = classif_pytorch(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1)
            acc_pt = (pred_pt == torch.tensor(y_test)).float().mean().item()
    acc_pt
    return (acc_pt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLP en keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X3_lang, keras, mo, y_lang):
    # Construction du perceptron en Keras
    model2 = Sequential([
        keras.layers.Input(shape=(3,)), #entree
        Dense(8, activation='tanh'), # couche cache [0] # tanh 加入非线性
        Dense(1, activation='sigmoid') # sortie : german or english ? -> 0 :genrman ; -> 1 : english
    ])

    # Compilation du modèle
    model2.compile(
        optimizer=SGD(), # SGD（随机梯度下降）
        loss='binary_crossentropy', # 二分类损失函数
        metrics=['accuracy'] # 训练的时候顺便告诉我准确率,只是为了观察模型
    )

    # Entraînement du modèle
    model2.fit(X3_lang, y_lang,
                        epochs=10, 
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2)

    # pseudo-Évaluation du modèle
    loss_keras2, acc_keras2 = model2.evaluate(X3_lang, y_lang, verbose=0)

    # Affichage des poids appris
    weights2 = model2.layers[0].get_weights()[0] # 把8个隐藏神经元这一层的 weights/bias拿出来看看
    bias2 = model2.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras2:.2%}\n\nPoids pour 'en_freq' : {weights2[0][0]:.4f}\n\nPoids pour 'th_freq' : {weights2[1][0]:.4f}\n\nBiais : {bias2[0]:.4f}")
    return acc_keras2, model2


@app.cell
def _(acc_keras2, acc_pt):
    (acc_keras2, acc_pt)
    return


@app.cell
def _(model2):
    model2.summary()
    return


if __name__ == "__main__":
    app.run()
