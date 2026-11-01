import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP3: classification en catégorie
    # convertir le notebook du TP2 sur le nouveau corpus ("tp3") de classification en rubriques et sources de presse

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
    import utils_tp3 as u

    return keras, mo, np, train_test_split, u


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
    corpus = u.load_text_json("./corpus-classif.json", limit=500)
    categories = list(sorted({ d['label'] for d in corpus }))
    code_categories = {c: i for i, c in enumerate(categories)}

    #langues = list(sorted({ d['label'] for d in corpus }))
    #code_langues = {l:i for i, l in enumerate(langues)}
    # u.normalise_data(dataset) # on le fait à l'intérieur du réseau
    return categories, code_categories, corpus


@app.cell
def _(np, train):
    lengths = [len(line['text'].split()) for line in train]
    print(np.percentile(lengths, [10, 50, 95, 100])) # pour choisir une longueur de vecteur
    return


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
    # train, test = train_test_split(corpus, train_size=0.8, random_state=42)
    # tv = keras.layers.TextVectorization(
    #         max_tokens=10,
    #         standardize=None,
    #         split="character",
    #         ngrams=(2,),
    #         output_mode="count",
    #     )

    train, test = train_test_split(corpus, train_size=0.8, random_state=42)
    tv = keras.layers.TextVectorization(
            max_tokens=1000,
            split="whitespace",
            ngrams=None,
            output_mode="count",
        )

    tv.adapt([line['text'] for line in train])
    return test, train, tv


@app.cell
def _(code_categories, np, test, train, tv):
    X_train = tv([line['text'] for line in train])
    Y_train = np.array([code_categories[line['label']] for line in train])

    X_test = tv([line['text'] for line in test])
    Y_test = np.array([code_categories[line['label']] for line in test])
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(keras, test, train):
    v_length = 80
    tv_int = keras.layers.TextVectorization(
        max_tokens=5000,
        split="whitespace",
        output_mode="int",
        output_sequence_length=v_length,
    )
    tv_int.adapt([line['text'] for line in train])

    X_train_int = tv_int([line['text'] for line in train])
    X_test_int = tv_int([line['text'] for line in test])
    print(X_train_int.shape)
    return X_test_int, X_train_int, tv_int, v_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    (voir dans le fichier annexe pour les détails)
    """)
    return


@app.cell
def _(X_train, Y_train, categories, tv, u):
    #model = u.model_sparse(tv.vocabulary_size(), len(categories))
    model = u.model_dropout(tv.vocabulary_size(), len(categories), rate=0.4) #avec dropout
    model.layers[0].adapt(X_train)
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)
    return history, model


@app.cell
def _(X_train_int, Y_train, categories, tv_int, u, v_length):
    model_emb = u.model_embedding(tv_int.vocabulary_size(), v_length, len(categories), pooling="max") # testé aussi avec pooling ="avg"
    history_emb = model_emb.fit(X_train_int, Y_train, epochs=30, batch_size=32, validation_split=0.2)
    return history_emb, model_emb


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


@app.cell
def _(X_test_int, Y_test, model_emb):
    model_emb.evaluate(X_test_int, Y_test)
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
        Y_test, # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1), 
        labels=range(len(categories)), 
        display_labels=categories
    )
    plt.show()
    return ConfusionMatrixDisplay, plt


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test_int,
    Y_test,
    categories,
    model_emb,
    np,
    plt,
):
    Y_pred_emb = model_emb.predict(X_test_int)

    ConfusionMatrixDisplay.from_predictions(
        Y_test,
        np.argmax(Y_pred_emb, axis=1),
        labels=range(len(categories)),
        display_labels=categories
    )
    plt.show()
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(model_emb):
    model_emb.summary()
    return


@app.cell
def _(tv):
    tv.get_vocabulary()
    return


@app.cell
def _(tv_int):
    tv_int.get_vocabulary()
    return


@app.cell
def _(history, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.gca()
    return


@app.cell
def _(history_emb, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(history_emb.history['accuracy'], label='Training Accuracy')
    plt.plot(history_emb.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs (Embedding)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
