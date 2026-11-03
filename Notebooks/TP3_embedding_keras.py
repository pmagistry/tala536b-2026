import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP3 : classification de textes de presse
    Nouvelle expérience : modèle Embedding simplifié, en conservant le corpus et le découpage du TP3.

    ## Imports usuels

    (ici j'utilise un fichier python annexe pour une partie de mon code, pensez à aller le voir aussi)
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp3_embedding as u

    return keras, mo, np, train_test_split, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lecture des données

    (fonctions issue du TP précédent, déplacées dans la bibliothèque)

    Le corpus JSON est lu par `u.load_json_text()`, qui reprend le format
    `label` / `text` attendu par le TP2. Les textes vides et les étiquettes
    vides sont ignorés.

    Question:
    - est-il possible de critiquer la normalisation du corpus à cet endroit ?
    """)
    return


@app.cell
def _(u):
    cible = "catégorie"  
    corpus = u.load_json_text("./corpus-classif.json", cible=cible)
    print(f"Cible : {cible} | Articles utilisables : {len(corpus)}")
    langues = list(sorted({ d['label'] for d in corpus }))
    print("Classes :", langues)
    code_langues = {l:i for i, l in enumerate(langues)}

    # u.normalise_data(dataset) # on le fait à l'intérieur du réseau
    return code_langues, corpus, langues


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage et encodage du corpus

    Le découpage train/validation/test est de 60 % / 20 % / 20 %.

    La vectorisation produit des séquences de 150 indices entiers. Pour cette expérience, seul le modèle change.

    Les étiquettes sont encodées en **sparse** (indices entiers).
    """)
    return


@app.cell
def _(corpus, keras, train_test_split):

    train_valid, test = train_test_split(
        corpus, train_size=0.8, random_state=42,
        stratify=[d["label"] for d in corpus]
    )
    train, valid = train_test_split(
        train_valid, train_size=0.75, random_state=42,
        stratify=[d["label"] for d in train_valid]
    )
    tv = keras.layers.TextVectorization(
            max_tokens=1000,
            standardize=None,
            split="whitespace",
            ngrams=None,
            output_mode="int",
            output_sequence_length=150,  # séquences de longueur fixe pour Embedding
        )      
    tv.adapt([line['text'] for line in train])
    return test, train, tv, valid


@app.cell
def _(code_langues, np, test, train, tv, valid):

    X_train = tv([line['text'] for line in train])
    Y_train = np.array([code_langues[line['label']] for line in train])

    X_test = tv([line['text'] for line in test])
    Y_test = np.array([code_langues[line['label']] for line in test])
    X_valid = tv([line['text'] for line in valid])
    Y_valid = np.array([code_langues[line['label']] for line in valid])
    return X_test, X_train, X_valid, Y_test, Y_train, Y_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    (voir dans le fichier annexe pour les détails)
    """)
    return


@app.cell
def _(X_train, X_valid, Y_train, Y_valid, langues, tv, u):

    pooling = "max"

    model = u.model_embedding_simple(
        vocab_size=tv.vocabulary_size(),
        seq_len=X_train.shape[1],
        nb_classes=len(langues),
        dim=32,
        pooling=pooling,
        rate=0.3
    )
    # Pas de Normalization().adapt() : les entrées sont des indices entiers.
    history = model.fit(
        X_train, Y_train, epochs=30, batch_size=32,
        validation_data=(X_valid, Y_valid)
    )
    return history, model


@app.cell
def _(history):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    return (plt,)


@app.cell
def _(history, plt):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return


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
def matrice_de_confusion(X_test, Y_test, langues, model, np, plt):
    from sklearn.metrics import ConfusionMatrixDisplay

    Y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        Y_test, # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1), 
        labels=range(len(langues)), 
        display_labels=langues
    )
    plt.show()
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(tv):
    tv.get_vocabulary()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
