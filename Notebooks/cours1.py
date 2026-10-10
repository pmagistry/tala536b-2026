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


@app.cell
def _(X_lin):
    print(X_lin)
    return


@app.cell
def _(y_lin):
    print(y_lin)
    return


@app.cell
def _(y_lin):
    y_lin[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Entraînement d'un perceptron (méthode classique)
    """)
    return


@app.cell
def _(X_lin, mo, torch, y_lin):
    def train_perceptron(X, y, eta=1.0, max_epochs=100):
        w = torch.zeros(X.shape[1]) # on crée une nouvelle matrice qui a la meme taille que la matrice x et qui est initialisé avec des 0.
        b = 0.0 #float (seuil theta?)
        for epoch in range(max_epochs):
            errors = 0 
            for i in range(X.shape[0]):
                xi, yi = X[i], y[i].item()
                yhat = 1.0 if (w @ xi + b) > 0 else 0.0 #produit cartesien = resultat    soit 1.0 | 0.0
                update = eta * (yi - yhat)   # dans {-1, 0, +1} eta = facteur amplificateur de l'erreur ?
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
def _(np, plt, w_perc):
    def plot_artifical_data(X, y, w, b, title="données et modèle linéaires"):
        fig, ax = plt.subplots(figsize=(5, 4))
        m = y == 1
        ax.scatter(X[m, 0], X[m, 1], c="tab:red", s=15, label="classe 1")
        ax.scatter(X[~m, 0], X[~m, 1], c="tab:blue", s=15, label="classe 0")
        xs = np.linspace(-3, 3, 50)
        if abs(w_perc[1]) > 1e-8:
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
    class MLP(nn.Module): #neuron network module  (  note : une classe c'est generaliser des attributs commun d'un objet)
        def __init__(self): #fonction qui s'execute lorsque je veux creer un objet de cette classe 
            super().__init__() # super se ref au parent( module) et execute la fonction initialisation 
            self.hidden = nn.Linear(2, 2) #couche cachée
            self.out = nn.Linear(2, 1)

        def forward(self, x):
            return self.out(torch.tanh(self.hidden(x)))

    mlp = MLP()

    print(list(mlp.parameters())) #demander au prof 

    # training in pytorch    poids         learning rate
    opt = torch.optim.SGD(mlp.parameters(), lr=0.5) #on appel la classe mlp et sgd 
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(5000):
        opt.zero_grad()  #gradient  qui sra utile pour ajuster les poids et reduire l'erreur
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
def _(mo, np):
    # chargement
    def load_data():
        data = []
        with open("./corpus/corpus6.txt") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th")
                n_en = text.count("en")
                n_le = text.count("le")
                n_os = text.count("os") #ajout pour detecter une troisieme langue
                l = len(text)
                instance = {"label":label, "th": n_th / l, "en": n_en /l, "le": n_le/l,"os":n_os/l}
                data.append(instance)
        return data
    import random

    classes = {"deu": 0, "eng": 1, "spa": 2}   #adaptation afin que le modele puisse corrctmeent laeliser les langues

    all_data = [d for d in load_data() if d['label'] in classes] 
    random.seed(0) #rend ce melange reproductible
    random.shuffle(all_data) #melange
    data_lang = all_data[:500]
    #pourquoi melanger les données? prsq il se peut que les 500 premiers textes ne concerne qu'une langue donc on ne pourrait pas identifier 3 langues mais apprendrait a reconnaitre une langue par rapport a d'autres.
    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k]- mean) / std # forcer la disctribution des count loi normal centré en 0 avec un std de 1

    normalise_data(data_lang)

    # encodage des données
    X_lang = np.array([[d['th'], d['en']] for d in data_lang])
    X3_lang = np.array([[d['th'], d['en'], d['le']] for d in data_lang])

    classes = {"deu": 0, "eng": 1, "spa": 2}   #pour les 3 langues

    X4_lang=np.array([[d['th'], d['en'], d['le'],d['os']] for d in data_lang])
    y_lang = np.array([0.0 if d['label'] == 'deu' else 1.0 for d in data_lang])
    y3_lang = np.array([classes[d['label']] for d in data_lang])
    mo.md("essayez avec et sans normalisation !")



    return X3_lang, X4_lang, X_lang, y3_lang, y_lang


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en Keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X_lang, keras, mo, y_lang):
    # Construction du perceptron en Keras
    model = Sequential([
        keras.layers.Input(shape=(2,)),
        Dense(8, activation='tanh'),
       # Dense(1,activation='softmax'),
        Dense(1,activation='sigmoid')
    
    ])



    # Compilation du modèle
    model.compile(
        optimizer=SGD(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model.fit(X_lang, y_lang,
                        #batch_size=32,
                        batch_size=16,  #essai de changer la taille des batch:taille des paquets dans lesquels on découpe les données d'entraînement ; chaque paquet donne lieu à une mise à jour des poids.
                        epochs=100, 
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2)

    # pseudo-Évaluation du modèle
    loss_keras, acc_keras = model.evaluate(X_lang, y_lang, verbose=0)

    # Affichage des poids appris
    weights = model.layers[0].get_weights()[0]
    bias = model.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras:.2%}\n\nPoids pour 'th_freq' : {weights[0][0]:.4f}\n\nPoids pour 'en_freq' : {weights[1][0]:.4f}\n\nBiais : {bias[0]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercice de classification en Pytorch
    """)
    return


@app.cell
def _(X3_lang, nn, torch, y_lang):
    def build_and_train_lang_classif(X, y):
        # building the "model"
        single_neuron = nn.Linear(3, 1)
        activation = nn.Sigmoid()

        # training
        opt = torch.optim.SGD(single_neuron.parameters(), lr=0.1)
        loss_fn = nn.BCELoss() #nn.BCEWithLogitsLoss()
        y_col = y.unsqueeze(1)

        losses = []                          
        for _ in range(100):
            opt.zero_grad()
            loss = loss_fn(activation(single_neuron(X)), y_col)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # pseudo-evaluation
        with torch.no_grad():
            acc = ((torch.sigmoid(single_neuron(X)) > 0.5).float() ==   y_col).float().mean().item()
        return single_neuron, acc,losses
    classif_pytorch, acc_pt, losses_pt = build_and_train_lang_classif(torch.tensor(X3_lang, dtype=torch.float32), torch.tensor(y_lang, dtype=torch.float32))
    return acc_pt, classif_pytorch, losses_pt


@app.cell
def _(acc_pt, classif_pytorch, losses_pt, mo, plt):
    fig_pt, ax_pt = plt.subplots(figsize=(5, 3))
    ax_pt.plot(losses_pt)
    ax_pt.set_xlabel("itération")
    ax_pt.set_ylabel("loss")
    ax_pt.set_title("Entraînement PyTorch")

    w_pt = classif_pytorch.weight.data.squeeze()
    b_pt = classif_pytorch.bias.data.item()

    mo.vstack([
        mo.md(f"""
    Précision PyTorch : {acc_pt:.2%}

    Poids pour 'th' : {w_pt[0]:.4f}  
    Poids pour 'en' : {w_pt[1]:.4f}  
    Poids pour 'le' : {w_pt[2]:.4f}  
    Biais : {b_pt:.4f}
    """),
        fig_pt
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##MLP Keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X3_lang, keras, mo, y_lang):
    # Construction du perceptron en Keras
    model2 = Sequential([
        keras.layers.Input(shape=(3,)),
        Dense(8,activation='tanh'),
        #Dense(5, activation='softmax'),
        Dense(1, activation='sigmoid'),
        #Dense(1, activation='sigmoid')
    ])

    # Compilation du modèle
    model2.compile(
        optimizer=SGD(),
        loss='binary_crossentropy',
        metrics=['accuracy']
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
    weights2 = model2.layers[0].get_weights()[0]
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Modele Keras pour plusieurs langues ?
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X4_lang, keras, mo, y3_lang):
    # Construction du perceptron en Keras
    model3 = Sequential([
        keras.layers.Input(shape=(4,)),  # 4features: fréquences de th, en, le, os
        Dense(8, activation='relu'), #couche caché plie l'espace pour rendre les    langues séparables linéairement
        Dense(3, activation='softmax'), #couche de sortie 1 neurone par langue, probas qui somment à 1
    ])

    model3.compile(
        optimizer=SGD(learning_rate=0.1), #descente de gradient, pas de 0.1
        loss='sparse_categorical_crossentropy', #N classes exclusives, labels entiers (0, 1, 2)
        metrics=['accuracy']
    )

    hist3 = model3.fit(X4_lang, y3_lang, 
                       epochs=50,           #nombre de passages complets sur les données
                       batch_size= 16,      #nb de correction de poid tout les 16 exemples
                       verbose=0,
                       shuffle=True,         #mélange les exemples à chaque epoch
                       validation_split=0.2) #Parti du corpus qui sera utilisé dans le test 
    #précision sur tout le corpus ( ce qu'a retenu le modele ),  puis sur la validation seule ( ce qu'a comprit le modele)
    loss_keras3, acc_keras3 = model3.evaluate(X4_lang, y3_lang, verbose=0)
    mo.md(f"Précision entraînement : {acc_keras3:.2%}\n\n"
          f"Précision validation : {hist3.history['val_accuracy'][-1]:.2%}")
    return (model3,)


@app.cell
def _(model3):
    model3.summary()
    return


@app.cell
def _():
    from collections import Counter
    with open("./corpus/corpus6.txt") as f:
        print(Counter(line.split(" ", 1)[0] for line in f))
    return


if __name__ == "__main__":
    app.run()
