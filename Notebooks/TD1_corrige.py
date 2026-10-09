import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    #import numpy as np
    import matplotlib.pyplot as plt
    # torch.manual_seed(0)
    return mo, nn, plt, torch


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
def _():
    from pathlib import Path
    import polars as pl
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Normalization
    from keras.optimizers import SGD

    #mo.md("une variable d'environnement vient définir le backend qui sera utilisé par Kera (torch, tensorflow ou jax)")
    return Dense, SGD, Sequential, keras


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chargement de données (disponibles sur Moodle)
    """)
    return


app._unparsable_cell(
    r"""
    # chargement
    chemin = "Notebooks/Corpus/corpus_2langue.txt"
    #chemin = "Notebooks/Corpus/corpus_6langues.txt"
    def load_data():
        data = []
        with open(chemin) as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th")
                n_en = text.count("en")
                n_le = text.count("le")
                l = len(text)
                instance = {"label":label, "th": n_th / l, "en": n_en /l, "le": n_le}
                data.append(instance)
        return data
    si t'es la 
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
    X_lang = np.array([[d['th'], d['en']] for d in data_lang])
    X3_lang = np.array([[d['th'], d['en'], d['le']] for d in data_lang])
    y_lang = np.array([0.0 if d['label'] == 'deu' else 1.0 for d in data_lang])
    mo.md("essayez avec et sans normalisation !")
    """,
    name="_"
)


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
        Dense(1, activation='sigmoid')
    ])

    # Compilation du modèle
    model.compile(
        optimizer=SGD(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model.fit(X_lang, y_lang,
                        epochs=10, 
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2)

    # pseudo-Évaluation du modèle
    loss_keras, acc_keras = model.evaluate(X_lang, y_lang, verbose=0)

    # Affichage des poids appris
    weights = model.layers[0].get_weights()[0]
    bias = model.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras:.2%}\n\nPoids pour 'en_freq' : {weights[0][0]:.4f}\n\nPoids pour 'th_freq' : {weights[1][0]:.4f}\n\nBiais : {bias[0]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercices
    ## classification de langues, version pytorch
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

        for _ in range(10):
            opt.zero_grad()
            loss = loss_fn(activation(single_neuron(X)), y_col)
            loss.backward()
            opt.step()

        # pseudo-evaluation
        with torch.no_grad():
            acc = ((torch.sigmoid(single_neuron(X)) > 0.5).float() == y_col).float().mean().item()
        return single_neuron, acc
    classif_pytorch, acc_pt = build_and_train_lang_classif(torch.tensor(X3_lang, dtype=torch.float32), torch.tensor(y_lang, dtype=torch.float32))
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
        keras.layers.Input(shape=(3,)),
        Dense(8, activation='tanh'),
        Dense(1, activation='sigmoid')
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
    ##Partie TD2 ajout du multilangue
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    note
    - le notebook ne veux pas que que je reutilise chemin et autre variable car deja existante
    - L'execution de l'entrainement prend énormément de temps. Environ 8 min (Peux être voir pour optimiser un aspect)
    """)
    return


@app.cell
def _():
    #Chargement corpsu 6 langue
    chemin_6lg = "Notebooks/Corpus/corpus_6langues.txt"
    def load_data_6lg(chemin):
        data = []
        with open(chemin) as file:
            for line in file: #Prend le doc par ligne du fichier 
                label, text = line.strip().split(" ", 1) #pour separer ph et code de la lang sinon les 2 ensembles
                n_th = text.count("th")
                n_en = text.count("en")
                n_le = text.count("le")
                l = len(text)
                instance = {"label": label, "th": n_th / l, "en": n_en / l, "le": n_le / l}
                data.append(instance)
        return data

    data_6lg = load_data_6lg(chemin_6lg)
    print(len(data_6lg))
    print(data_6lg)
    return (data_6lg,)


@app.cell
def _(data_6lg):
    #Séparation du corpus
    from sklearn.model_selection import train_test_split

    labels_6lg = [] #pour recup les langues
    for text in data_6lg:
        langue = text["label"]
        labels_6lg.append(langue)

    res = train_test_split( #Perme de decouper le coursp en train et le reste 70/30
        data_6lg,
        test_size=0.3,
        stratify=labels_6lg,
        random_state=0,
    )
    entrainement = res[0]
    reste = res[1]

    labels_reste = [] #recuperer pour le reste
    for txt in reste:
        langue = txt["label"]
        labels_reste.append(langue)

    res2 = train_test_split(
        reste,
        test_size=0.5,
        stratify=labels_reste,
        random_state=0,
    )
    verif = res2[0]
    teste = res2[1]

    print ('Corpus Train:',len(entrainement))
    print ('Corpus Check:',len(verif))
    print ('Corpus Teste:',len(teste))
    return entrainement, teste, verif


@app.cell
def _(entrainement):
    #Calcule de l écart type et de la moyenne sur sous corpsu entrainement
    import numpy as np #a commenter si cellule d'avant lancer

    cles_6lg = ["th", "en", "le"]

    moyennes_6lg = {}
    ecarts_types_6lg = {}

    for c in cles_6lg: #3 features traiter 1 a 1
        v = []
        for texte in entrainement:
            val = texte[c]
            v.append(val)
        moyenne = np.mean(v)
        ecart_type = np.std(v)
        moyennes_6lg[c] = moyenne
        ecarts_types_6lg[c] = ecart_type

    print("moyenne : ", moyennes_6lg)
    print("ecart type : ", ecarts_types_6lg)
    return cles_6lg, ecarts_types_6lg, moyennes_6lg, np


@app.cell
def _(cles_6lg, ecarts_types_6lg, entrainement, moyennes_6lg, teste, verif):
    #Normalisation
    def normalise_liste(dataset, moyennes, ecarts_types, cles):
        nouveau_dataset = []
        for doc in dataset:
            nouveau_doc = {}
            nouveau_doc["label"] = doc["label"]
            for cle in cles:
                valeur = doc[cle]
                moyenne = moyennes[cle]
                ecart_type = ecarts_types[cle]
                valeur_normalisee = (valeur - moyenne) / ecart_type
                nouveau_doc[cle] = valeur_normalisee
            nouveau_dataset.append(nouveau_doc)
        return nouveau_dataset

    entrainement_n = normalise_liste(entrainement, moyennes_6lg, ecarts_types_6lg, cles_6lg)
    verif_n = normalise_liste(verif, moyennes_6lg, ecarts_types_6lg, cles_6lg)
    teste_n = normalise_liste(teste, moyennes_6lg, ecarts_types_6lg, cles_6lg)
 
    print (len(entrainement_n))
    print (len(verif_n))
    print (len(teste_n))
    return entrainement_n, teste_n, verif_n


@app.cell
def _(cles_6lg, entrainement_n, np, teste_n, verif_n):
    #Prépa tableau poru keras
    langue_vers_entier = {
        "deu": 0,
        "eng": 1,
        "fra": 2,
        "ita": 3,
        "nld": 4,
        "spa": 5,
    }

    def extraire_x(dataset, cles): #met les liste des res en tab
        liste_x = []
        for doc in dataset:
            ligne = []
            for cle in cles:
                ligne.append(doc[cle])
            liste_x.append(ligne)
        return np.array(liste_x, dtype="float32") #besoin de préciser pour eviter des erreurs

    def extraire_y(dataset, dico_langues):
        liste_y = []
        for doc in dataset:
            liste_y.append(dico_langues[doc["label"]])
        return np.array(liste_y, dtype="int32") #pas float pour eviter anbiguité et car keras attend ça pour les class

    #Application au 3 sous corpus
    X_entrainement = extraire_x(entrainement_n, cles_6lg)
    y_entrainement = extraire_y(entrainement_n, langue_vers_entier)

    X_verif = extraire_x(verif_n, cles_6lg)
    y_verif = extraire_y(verif_n, langue_vers_entier)

    X_teste = extraire_x(teste_n, cles_6lg)
    y_teste = extraire_y(teste_n, langue_vers_entier)

    print(X_entrainement.shape, y_entrainement.shape)
    print(X_verif.shape, y_verif.shape)
    print(X_teste.shape, y_teste.shape)
    return X_entrainement, X_teste, X_verif, y_entrainement, y_teste, y_verif


@app.cell
def _(X_entrainement, X_verif, keras, y_entrainement, y_verif):
    #utilisation de keras pour l'apprentissage
    modele_6lg = keras.Sequential()


    couche_entree = keras.Input(shape=(3,)) #3 car 3 feat th le en
    modele_6lg.add(couche_entree)

    couche_cachee = keras.layers.Dense(8, activation="tanh") #8 neuronne et copie de TP1
    modele_6lg.add(couche_cachee)

    couche_sortie = keras.layers.Dense(6, activation="softmax") #6 neuronne car 6 langue 
    modele_6lg.add(couche_sortie)


    optimiseur_6lg = keras.optimizers.SGD(learning_rate=0.1) #La lenteur du modèle vient surment de là
    modele_6lg.compile(
        optimizer=optimiseur_6lg,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    historique_6lg = modele_6lg.fit(
        X_entrainement,
        y_entrainement,
        epochs=20,
        batch_size=32,
        shuffle=True,
        validation_data=(X_verif, y_verif),
        verbose=1,
    )
    return (modele_6lg,)


@app.cell
def _(X_teste, modele_6lg, y_teste):
    resultat_test = modele_6lg.evaluate(X_teste, y_teste, verbose=0)

    loss_test = resultat_test[0]
    accuracy_test = resultat_test[1]

    print(f" Évaluation finale sur le jeu de test : {len(y_teste)} documents")

    print(f"Loss test : {loss_test:.3f}")
    print(f" Accuracy test : {accuracy_test:.3f}")
    return (resultat_test,)


@app.cell
def _(X_entrainement, X_verif, keras, y_entrainement, y_verif):
    #teste avec beaucoup plus de neuronne pour comprer resulats
    modele_6lg_b = keras.Sequential()

    modele_6lg_b.add(keras.Input(shape=(3,)))
    modele_6lg_b.add(keras.layers.Dense(16, activation="tanh"))
    modele_6lg_b.add(keras.layers.Dense(16, activation="tanh"))
    modele_6lg_b.add(keras.layers.Dense(6, activation="softmax"))

    optimiseur_6lg_b = keras.optimizers.SGD(learning_rate=0.1)

    modele_6lg_b.compile(
        optimizer=optimiseur_6lg_b,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    historique_6lg_b = modele_6lg_b.fit(
        X_entrainement,
        y_entrainement,
        epochs=20,
        batch_size=32,
        shuffle=True,
        validation_data=(X_verif, y_verif),
        verbose=1,
    )
    return (modele_6lg_b,)


@app.cell
def _(X_teste, modele_6lg_b, resultat_test, y_teste):
    resultat_test_b = modele_6lg_b.evaluate(X_teste, y_teste, verbose=0)

    loss_test_b = resultat_test[0]
    accuracy_test_b = resultat_test[1]

    print(f" Évaluation finale sur le jeu de test : {len(y_teste)} documents")

    print(f"Loss test : {loss_test_b:.3f}")
    print(f" Accuracy test : {accuracy_test_b:.3f}")
    return


@app.cell
def _(X_entrainement, X_verif, keras, y_entrainement, y_verif):
    #teste avec moin de neuronnes pour comprer resulats
    modele_6lg_c = keras.Sequential()

    modele_6lg_c.add(keras.Input(shape=(3,)))
    modele_6lg_c.add(keras.layers.Dense(4, activation="tanh"))
    modele_6lg_c.add(keras.layers.Dense(6, activation="softmax"))

    optimiseur_6lg_c = keras.optimizers.SGD(learning_rate=0.1)

    modele_6lg_c.compile(
        optimizer=optimiseur_6lg_c,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    historique_6lg_c = modele_6lg_c.fit(
        X_entrainement,
        y_entrainement,
        epochs=15,
        batch_size=128,
        shuffle=True,
        validation_data=(X_verif, y_verif),
        verbose=1,
    )
    return (modele_6lg_c,)


@app.cell
def _(X_teste, modele_6lg_c, resultat_test, y_teste):
    resultat_test_v = modele_6lg_c.evaluate(X_teste, y_teste, verbose=0)

    loss_test_c = resultat_test[0]
    accuracy_test_c = resultat_test[1]

    print(f" Évaluation finale sur le jeu de test : {len(y_teste)} documents")

    print(f"Loss test : {loss_test_c:.3f}")
    print(f" Accuracy test : {accuracy_test_c:.3f}")
    return


if __name__ == "__main__":
    app.run()
