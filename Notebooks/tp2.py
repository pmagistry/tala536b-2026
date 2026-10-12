import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    # torch.manual_seed(0)
    return nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo, torch):
    N = 100
    X_lin = torch.randn(N, 2)
    y_lin = (X_lin[:, 0] + 2 * X_lin[:, 1] > 0).float()
    mo.md("Données artificielles, linéairement séparables dans R^2")
    return X_lin, y_lin


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    J'ai sélectionné les 10 bigrammes les plus fréquents pour chaque langue à partir des données d'entraînement. Après suppression des doublons, j'ai obtenu 31 caractéristiques.

    J'ai ensuite entraîné un perceptron multiclasses avec 6 neurones de sortie et une activation softmax. Après 10 époques, le modèle atteint une précision de 96,24 % sur les données de test.
    """)
    return


@app.cell
def _():

    # chargement de données
    def load_data():
        data = [] # data is a list
        with open("./corpus6.txt", encoding="utf-8") as file:
            for line in file:
                label, text = line.strip().split(" ",1) # 1 shows how many times we are going to split. since the given data has always a label before the first space, this is to split the text with its label. 
                instance = {
                    "label": label,
                    "text": text
                } # on a créé un dictionnaire avec le label et le texte.
                data.append(instance) # on ajoute le dict à notre liste.
        return data

    data_lang = load_data()
    langues = ['deu', 'eng', 'fra', 'ita', 'nld', 'spa']
    return data_lang, langues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Séparation pour avoir une partie à train et un autre pour test
    """)
    return


@app.cell
def _(data_lang):
    from sklearn.model_selection import train_test_split #on utilise train test split pour diviser les textes en test et train

    train_data, test_data = train_test_split(
        data_lang, #notre donnée de texte
        test_size=0.20, #20% pour les test
        random_state=42, # pour qu'on peut repeter la meme partition après
        stratify=[d["label"] for d in data_lang] # pour avoir proche pourcentage pour chaque langue
    )
    return test_data, train_data


@app.cell
def _(data_lang, test_data, train_data):
    print("Total:", len(data_lang)) # pour voir combien de texte
    print("Train:", len(train_data)) # pour voir combien de texte pour train
    print("Test:", len(test_data)) # pour voir combien de txte pour teste

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cherchons les plus fréquentes bigrammes
    """)
    return


@app.cell
def _():
    import re
    from collections import Counter

    def compteur_bigrammes(text):
        words = re.findall(r"[^\W\d_]+", text.lower()) # on enleve les nombres, ponctuation
        bigrammes = [] # on crée notre liste

        for word in words:
            for i in range(len(word) - 1): # on ne va pas jusqu'à la derniere lettre comme c'est pas possible d'avoir une autre lettre après lui
                bigrammes.append(word[i:i+2]) # on ajoute une lettre et la lettre qui vient après
        return bigrammes

    #print(compteur_bigrammes("The cat!")) # pour tester
    return Counter, compteur_bigrammes


@app.cell
def _(Counter, compteur_bigrammes, langues, train_data):
    frequences = {}

    for langue in langues:
        compteur = Counter()

        for d in train_data:
            if d["label"] == langue:
                compteur.update(compteur_bigrammes(d["text"]))

        frequences[langue] = compteur.most_common(10)
        
    for langue, bigrammes in frequences.items():
        print(langue, bigrammes)
    return (frequences,)


@app.cell
def _(frequences):
    bigrammes_choisis = sorted({
        bigramme
        for liste in frequences.values()
        for bigramme, nombre in liste
    })

    print("Nombre de biagremmes uniques:", len(bigrammes_choisis))
    print(bigrammes_choisis)
    return (bigrammes_choisis,)


@app.cell
def _(
    Counter,
    bigrammes_choisis,
    compteur_bigrammes,
    np,
    test_data,
    train_data,
):
    def vectoriser_textes(dataset, bigrammes):
        X = []

        for d in dataset:
            compteur = Counter(compteur_bigrammes(d["text"])) # on compte la frequence de chaque bigramme
            total = sum(compteur.values()) # sum de tous les bigrammes

            vecteur = []

            for bigramme in bigrammes: 
                if total > 0: # pour éviter les problems s'il y a des lignes avec une seule lettre
                    frequence = compteur[bigramme] / total
                else:
                    frequence = 0
                
                vecteur.append(frequence)
            
            X.append(vecteur)
        
        return np.array(X, dtype=np.float32)

    X_train = vectoriser_textes(train_data, bigrammes_choisis) # on vectorise les texte trains
    X_test = vectoriser_textes(test_data, bigrammes_choisis) # on vectorise les texte tests

    print("Train:", X_train.shape) # pour voir combien de lignes et bigrammes
    print("Test:", X_test.shape)
    return X_test, X_train


@app.cell
def _(langues, np, test_data, train_data):
    # encodage des étiquettes
    y_train = []

    for exemple in train_data:
        label = exemple["label"] #fra
        index = langues.index(label) #2
        y_train.append(index) # on ajout 2

    y_train = np.array(y_train)

    y_test = []

    for exemple in test_data:
        label = exemple["label"]
        index = langues.index(label)
        y_test.append(index)

    y_test = np.array(y_test)

    print("Train étiquettes:", y_train.shape)
    print("Test étiquettes:", y_test.shape)
    return y_test, y_train


@app.cell
def _(X_test, X_train):
    # normalisation
    moyenne = X_train.mean(axis=0) # calcule de moyenne
    ecart_type = X_train.std(axis=0) # calcule de l'ecart type

    ecart_type[ecart_type == 0] = 1 # pour éviter la division à 0

    X_train_norm = (X_train - moyenne) / ecart_type
    X_test_norm = (X_test - moyenne) / ecart_type

    print(X_train_norm.shape)
    print(X_test_norm.shape)
    return X_test_norm, X_train_norm


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
    X_test_norm,
    X_train_norm,
    keras,
    y_test,
    y_train,
):
    # Construction du perceptron en Keras
    model = Sequential([
        keras.layers.Input(shape=(31,)),
        Dense(6, activation='softmax')
    ])

    # Compilation du modèle
    model.compile(
        optimizer=SGD(),
        loss='sparse_categorical_crossentropy', # pour classifier en plusieurs categories
        metrics=['accuracy']
    )

    # Entraînement du modèle
    history = model.fit(
        X_train_norm,
        y_train,
        epochs=10, 
        verbose=1, 
        shuffle=True, 
        validation_split=0.2
    )

    # Évaluation sur les données de test
    loss_keras, acc_keras = model.evaluate(
        X_test_norm,
        y_test,
        verbose=0
    )

    print(f"Précision sur le test : {acc_keras:.2%}")
    print(f"Loss sur le test : {loss_keras:.4f}")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Epoch 1/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8877 - loss: 0.4008 - val_accuracy: 0.9464 - val_loss: 0.2092

    Epoch 2/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9517 - loss: 0.1840 - val_accuracy: 0.9548 - val_loss: 0.1680

    Epoch 3/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9567 - loss: 0.1585 - val_accuracy: 0.9572 - val_loss: 0.1525

    Epoch 4/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9589 - loss: 0.1469 - val_accuracy: 0.9586 - val_loss: 0.1440

    Epoch 5/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9600 - loss: 0.1402 - val_accuracy: 0.9593 - val_loss: 0.1387

    Epoch 6/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9610 - loss: 0.1356 - val_accuracy: 0.9595 - val_loss: 0.1350

    Epoch 7/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9616 - loss: 0.1324 - val_accuracy: 0.9603 - val_loss: 0.1323

    Epoch 8/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9622 - loss: 0.1300 - val_accuracy: 0.9605 - val_loss: 0.1302

    Epoch 9/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9627 - loss: 0.1280 - val_accuracy: 0.9611 - val_loss: 0.1285

    Epoch 10/10
    1923/1923 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9629 - loss: 0.1265 - val_accuracy: 0.9610 - val_loss: 0.1271

    Précision sur le test : 96.24%

    Loss sur le test : 0.1258
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrice de confusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pour la matrice de confusion, j'ai demandé de l'aide à ChatGPT car je ne savais pas encore comment la visualiser avec Matplotlib. J'ai ensuite utilisé le graphique pour analyser les erreurs de classification du modèle.
    """)
    return


@app.cell
def _(X_test_norm, langues, model, np, plt, y_test):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    predictions = model.predict(X_test_norm, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=langues
    )

    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion — Classification en 6 langues")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Résultats et évaluation

    Le modèle utilise 31 bigrammes sélectionnés à partir des données d'entraînement.

    Après 10 époques, le modèle obtient une précision de 96,24 % sur les données de test, avec une loss de 0,1258.

    La matrice de confusion montre que les principales erreurs concernent le français et l'italien, ainsi que l'allemand et le néerlandais.

    Pour la visualisation de la matrice de confusion, j'ai utilisé ChatGPT.
    """)
    return


if __name__ == "__main__":
    app.run()
