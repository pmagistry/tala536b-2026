import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP2 : classification de texte
    L'objectif est de tester le changement de loss et de produire un classifieur de textes d'articles de presse.

    ## Imports usuels

    (ici j'utilise un fichier python annexe pour une partie de mon code, pensez à aller le voir aussi)
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import os

    from sklearn.model_selection import train_test_split
    import utils_tp2 as u

    return mo, np, os, train_test_split, u


@app.cell
def _(os):
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Normalization
    from keras.optimizers import SGD

    return Dense, Sequential, keras


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lecture des données

    (fonctions issue du TP précédent, déplacées dans la bibliothèque)

    Ce code peut aussi marcher avec le corpus6.txt

    Question:
    - est-il possible de critiquer la normalisation du corpus à cet endroit ?
    """)
    return


@app.cell
def _(u):
    # corpus = u.load_text("./corpus6.txt", limit=500)
    corpus = u.load_json("./corpora/corpus-classif.json", limit=500)
    categories = list(sorted({d["cat"] for d in corpus}))
    code_categories = {l: i for i, l in enumerate(categories)}
    print(categories)
    # u.normalise_data(dataset) # on ne le fait pas avec de la classification de texte, la couche TextVectorization s'occupe de la conversion en vecteurs et le problème n'est plus un problème de valeurs numériques comme avant
    return categories, code_categories, corpus


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage et encodage du corpus

    Le découpage 80/20 est réalisé avec scikit-learn

    L'encodage se fait ici en ***one-hot***, une version **sparse** des fonctions et du modèle sont disponibles
    """)
    return


@app.cell
def _(corpus, keras, train_test_split):
    train_validation, test = train_test_split(corpus, train_size=0.9, random_state=42)
    # on rajoute un découpage pour avoir un corpus de validation différent du corpus de test
    train, validation = train_test_split(
        train_validation, train_size=0.9, random_state=42
    )
    print(corpus[:10])
    seq_len = 300
    tv = keras.layers.TextVectorization(
        max_tokens=500,
        # standardize=None,
        split="whitespace",
        ngrams=None,  # (2,),
        output_mode="int",
        output_sequence_length=seq_len,  # taille en tokens des textes vectorisés
    )
    tv.adapt([line["text"] for line in train])
    return seq_len, test, train, tv, validation


@app.cell
def _(code_categories, np, test, train, tv, validation):
    X_train = tv([line["text"] for line in train])
    Y_train = np.array([code_categories[line["cat"]] for line in train])

    X_test = tv([line["text"] for line in test])
    Y_test = np.array([code_categories[line["cat"]] for line in test])

    X_validation = tv([line["text"] for line in validation])
    Y_validation = np.array([code_categories[line["cat"]] for line in validation])
    return X_test, X_train, X_validation, Y_test, Y_train, Y_validation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    (voir dans le fichier annexe pour les détails)
    """)
    return


@app.cell
def _(
    Dense,
    Sequential,
    X_train,
    X_validation,
    Y_train,
    Y_validation,
    categories,
    keras,
    seq_len,
):
    model = Sequential(
        [
            keras.layers.Input(shape=(seq_len,)),
            keras.layers.Embedding(input_dim=seq_len, output_dim=32, mask_zero=True),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dropout(0.8),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(len(categories), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_validation, Y_validation),
    )
    return history, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Évaluation sur le jeu de test

    Question:
    - Serait-il pertinent d'effectuer le split_valid au moment de l'entraînement ? Quelle différence ?
    """)
    return


@app.cell
def _(X_test, Y_test, model):
    model.evaluate(X_test, Y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrice de confusion

    Un point important à comprendre ici est le rôle des np.argmax

    - Que se passe-t-il si on passe en encodage ***sparse*** ?
    """)
    return


@app.cell
def matrice_de_confusion(X_test, Y_test, categories, model, np):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    Y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        Y_test,  # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1),
        labels=range(len(categories)),
        display_labels=categories,
    )
    plt.show()
    return (plt,)


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(tv):
    tv.get_vocabulary()
    return


@app.cell
def _(history, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
