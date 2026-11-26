import marimo

__generated_with = "0.25.2"
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

    classe 1 si

    $$ 2y + x  > 0 $$

    sinon classe 0.

    /// warning | Correction
    <small>Version originale : *« classe 0 si $2y + x > 0$, sinon classe 1 »*.</small>

    Le texte original inversait les classes par rapport au code.
    `y_lin = (X_lin[:, 0] + 2 * X_lin[:, 1] > 0).float()` renvoie **1**
    quand $x + 2y > 0$ (`True` → `1.0`), donc ces points sont en **classe 1**.
    C'est le code qui fabrique les étiquettes : c'est lui qui fait foi.
    ///

    Autrement dit on sépare les données par la droite :
    $$ y = -\frac{x}{2} $$
    """)
    return


@app.cell
def _(mo, torch):

    N = 100
    X_lin = torch.randn(N, 2) # entrée random
    y_lin = (X_lin[:, 0] + 2 * X_lin[:, 1] > 0).float() # résultat/étiquette d'appartenance à la classe 0 ou 1(classe de chaque point)
    mo.md("Données artificielles, linéairement séparables dans R^2")

    print(X_lin[:10])
    print(y_lin[:10])

    print(len(X_lin))
    print(len(y_lin))
    return X_lin, y_lin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Entraînement d'un perceptron (méthode classique)
    """)
    return


@app.cell
def _(X_lin, mo, torch, y_lin):
    def train_perceptron(X: torch.Tensor, y: torch.Tensor, eta=1.0, max_epochs=100):
        # torch.zeros() -> génére un tenseur contenant 0.0
        # X.shape[1] -> donne le nombre de colonne (shape[0] nb de ligne)
        # w = tenseur à une dimension de  longueur 2 (X.shape[1])
        # chaque entrée de w représente le poids associé à une colonne
        w = torch.zeros(X.shape[1]) # tenseurs de 2 nombres
        # print(len(X))
        # print(len(y))
        # print(w)
        b = 0.0 #biais
        for epoch in range(max_epochs): # jusqu'à 100 époques max (époque = 1 passage sur tout X)
            errors = 0
            for i in range(X.shape[0]): # pour chaque ligne 
                xi, yi = X[i], y[i].item() # .item() : tenseur -> float
                #if (i==0): 
                    #print(xi)
                # @ produit scalaire des poids avec une ligne des données 
                yhat = 1.0 if (w @ xi + b) > 0 else 0.0 # yhat = ŷ -> prédiction (0 ou 1 )
                # eta = taux d'apprentissage ou pas 
                update = eta * (yi - yhat)   # dans {-1, 0, +1}
                # update = eta * (vérité-prédiction)
                if update != 0:
                    w += update * xi
                    b += update
                    #print(w,b)
                    errors += 1
            if errors == 0:
                print("plus de fautes c'est bon")
                print("nombre d'epoch",epoch + 1) # epoch commence à 0
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
        if abs(w[1]) > 1e-8: # corrigé : testait w_perc[1] (le perceptron) au lieu du w passé en paramètre
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
        single_neuron = nn.Linear(2, 1) #2 caractéristiques, 1 neurone qui regarde 2 caractéristiques
        # Linear = calcule w·x + b, donc la frontière est une droite ; ça suffit ici car les données sont linéairement séparables
        # seul l'apprentissage change; la méthode de classification ne change pas 

        # training
        opt = torch.optim.SGD(single_neuron.parameters(), lr=0.1) 
        # optim = module qui contient les optimiseurs càd les algo qui mettent à jour les poids
        # on utilise l'algo SGD qui applique la règle w <- w-lr*gradients
        # ici la descente n'est pas stochastique : on passe les 100 points à chaque pas (full batch). Avec des mini-batchs, le même SGD deviendrait stochastique
        # il garde une référence vers w et b (càd les paramètres du neurone), il les modifiera ensuite ce qui fera évoluer le neurone

        loss_fn = nn.BCEWithLogitsLoss() # l'évaluateur; ilprend le score brut et le transformera en une proba (chiffre entre 0 et 1) grace à la sigmoide; il surveillera les prédictions du neurone et comparera ses prédictions avec la "vérité" (dans notre cas y)
        # dit autrement "la fonction qui prend les scores bruts du neurone, les transforme elle-même en probabilités avec la sigmoïde, puis mesure à quel point ces probabilités se trompent"
        # si c'était BCELoss il faudrait appliquer la fonction de sigmoid à single_neuron(X); donc faire sigmoid(single_neuron(X)); la on s'épargne une étape
        y_col = y.unsqueeze(1) # au lieu d'avoir tensor([1., 0., 0., ...])
        # on a tensor([[1.][0.][0.]...])
        # c'est juste pour que les données soient alignées

        for _ in range(500): # 500 époches avec un batch de 100
            opt.zero_grad() # correction adaptée à l'epoch actuelle; remet .grad à zéro (efface le tableau) ; sinon backward additionne le nouveau gradient aux anciens. Les poids, eux, gardent ce qu'ils ont appris
            loss = loss_fn(single_neuron(X), y_col) # on met devant l'évaluateur le petit à entrainer avec la feuille des résultats attendus; le jury donne ensuite sa note
            loss.backward() # parcourt à l'envers les opérations enregistrées (rétropropagation) et calcule le gradient (p − y)·x, rangé dans w.grad et b.grad. Ne modifie pas encore les poids
            # backward donne la direction de la correction (sans lr ni signe moins) ; c'est step qui en fait l'update
            opt.step() # c'est ici qu'intervient la maj w <- w-lr*w.gradient
            # équivalent SGD du w += update * xi du perceptron : même idée (corriger w dans la direction de x), mais erreur continue (p − y) et gradient soustrait

        # pseudo-évaluation : on teste sur les données d'entraînement. En vrai, on garderait une partie des données (test) cachée pendant l'entraînement
        with torch.no_grad(): # examen, fini l'enregistrement on évalue simplement les capacités entrainées
            #PyTorch arrête de retenir les opérations, ce qui économise de la mémoire et du temps.
            acc = ((torch.sigmoid(single_neuron(X)) > 0.5).float() == y_col).float().mean().item() # le neuron fait sa classification binaire (le seuil est à 0.5) des 100 points et on compare avec la "vérité"
            # .float().mean() on calcule la proportion de bonne réponse
            # .item() on récupère le nombre
        return single_neuron, acc # on renvoie le neurone entrainé et son % de bonne réponse 
    single, acc = build_and_train_single_neuron(X_lin, y_lin) #on donne toujours les 2 tenseurs, un avec les points aléatoire, l'autre avec l'appartenance à la classe 0 ou 1 
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
            self.hidden = nn.Linear(2, 2) # couche cachée avec deux neurones 
            self.out = nn.Linear(2, 1) # couche extérieur avec un seul neurone qui prendra en entrée les h (sortie de la couche cachée)

        def forward(self, x): 
            #calcul de la couche cachée -> de tanh -> de la couche extérieure
            return self.out(torch.tanh(self.hidden(x))) 

    mlp = MLP() # initialise l'objet MLP

    # training in pytorch
    opt = torch.optim.SGD(mlp.parameters(), lr=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(5000): # même fonctionnement que le SGD 
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


@app.cell
def _():
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
        with open("./../../data/corpus.txt") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th") # compte bigram
                n_en = text.count("en")
                l = len(text)
                instance = {"label":label, "th": n_th / l, "en": n_en /l}
                data.append(instance)
        return data

    data_lang = load_data()[:200]

    # normalisation
    # x_normalisé = (x − moyenne) / écart-type 
    def normalise_data(dataset):
        for k in dataset[0].keys(): 
            if k != 'label':
                mean = np.mean([d[k] for d in dataset]) # moyenne de cette caractéristique sur toutes les phrases
                std = np.std([d[k] for d in dataset]) # écart-type de cette caractéristique (standart deviation)
                for d in dataset:
                    d[k] = (d[k]- mean) / std

    # normalisation : remet th et en à l'échelle ~1. Sans elle, x ≈ 0,01 donc le gradient (p − y)·x
    # est ~100× trop petit : les poids n'apprennent presque pas (ce n'est pas du vanishing gradient,
    # qui vient de la saturation de l'activation dans les réseaux profonds)
    print(data_lang)
    normalise_data(data_lang)
    print(data_lang)
    # encodage des données
    X_lang = np.array([[d['th'], d['en']] for d in data_lang])
    y_lang = np.array([0.0 if d['label'] == 'deu' else 1.0 for d in data_lang])

    print(X_lang[:10])
    print(y_lang[:10])
    mo.md("essayez avec et sans normalisation !")
    return X_lang, y_lang


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
        keras.layers.Input(shape=(2,)), # 2 caractéristiques (th,en)
        Dense(4, activation='tanh'),
        Dense(1, activation='sigmoid') # 1 neurone, puis sigmoide
    ])

    # Compilation du modèle
    model.compile(
        optimizer=SGD(), # selection dans le catalogues de l'algo SGD
        loss='binary_crossentropy', # choix de BCE, pas besoin de logits prcq on a deja la sigmoid sur le model
        metrics=['accuracy'] 
    )

    # Entraînement du modèle
    model.fit(X_lang, y_lang,
                        epochs=70, # après 140 epoch, 95% de précision avec normalisation
                          # sans normalisation : bloqué à ~55 % même avec 700 epochs. Les entrées minuscules donnent un gradient minuscule,
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2) # garde 20% des données pour la validation
    # par défaut keras fait des mini-batchs de 32 pas comme pytorch  

    # Précision du modèle : 55.50% Poids pour 'th_freq' : -0.8248 Poids pour 'en_freq' : 0.1748
    # seul le biais apprend → le modèle répond toujours "deu" (111/200 = 55,5 %, la classe majoritaire)

    # pseudo-Évaluation du modèle
    loss_keras, acc_keras = model.evaluate(X_lang, y_lang, verbose=0)

    # Affichage des poids appris
    weights = model.layers[0].get_weights()[0]
    bias = model.layers[0].get_weights()[1]
    # corrigé : les étiquettes étaient inversées ; X_lang = [th, en] donc la colonne 0 est th
    mo.md(f"Précision du modèle : {acc_keras:.2%}\n\nPoids pour 'th_freq' : {weights[0][0]:.4f}\n\nPoids pour 'en_freq' : {weights[1][0]:.4f}\n\nBiais : {bias[0]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification en Pytorch
    """)
    return


@app.cell
def _(X_lang, mo, nn, torch, y_lang):

    from torch.utils.data import TensorDataset, DataLoader
    # construction et entraînement d'un neurone en Pytorch (équivalent du modèle Keras)

    def build_and_train_neuron(x, y):
        tx = torch.tensor(x, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.float32)
        ty_col = ty.unsqueeze(1)

        n_train = int(0.8 * len(tx))
        tx_train, tx_val = tx[:n_train], tx[n_train:]
        ty_col_train, ty_col_val = ty_col[:n_train], ty_col[n_train:]

        dataset_train = TensorDataset(tx_train, ty_col_train)
        loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

        neuron = nn.Linear(2, 1)
        opt = torch.optim.SGD(neuron.parameters(), lr=0.1)
        loss_fn = nn.BCEWithLogitsLoss()

        # entrainement
        for _ in range(70): # epoch
            for xb, yb in loader: # mini-batch de 32 points
                opt.zero_grad()
                loss = loss_fn(neuron(xb), yb)
                loss.backward()
                opt.step()

        # évaluation
        with torch.no_grad():
            # corrigé : tx_val (et non tx_col_val), seuil > 0.5 avant la comparaison, puis moyenne des bonnes réponses
            acc_train = ((torch.sigmoid(neuron(tx_train)) > 0.5).float() == ty_col_train).float().mean().item()
            acc_val = ((torch.sigmoid(neuron(tx_val)) > 0.5).float() == ty_col_val).float().mean().item() 

        return neuron, acc_train, acc_val 

    neuron_pt, acc_train_pt, acc_val_pt = build_and_train_neuron(X_lang, y_lang) 
    mo.md(f"Précision entraînement : {acc_train_pt:.2%}\n\nPrécision validation : {acc_val_pt:.2%}\n\nPoids pour 'th_freq' : {neuron_pt.weight[0][0].item():.4f}\n\nPoids pour 'en_freq' : {neuron_pt.weight[0][1].item():.4f}\n\nBiais : {neuron_pt.bias[0].item():.4f}")
    return


if __name__ == "__main__":
    app.run()
