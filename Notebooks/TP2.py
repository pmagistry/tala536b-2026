import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP2 : classification en 6 langues
    L'objectif est de tester le changement de loss, l'activation softmax et de produire un classifieur pour plus de deux langues.

    ## Imports usuels

    (ici j'utilise un fichier python annexe pour une partie de mon code, pensez à aller le voir aussi)
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import json
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp2 as u

    return keras, mo, np, plt, train_test_split, u


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
    corpus = u.load_json("./corpus/corpus-classif.json",)
    #langues = list(sorted({ d['label'] for d in corpus }))
    #code_langues = {l:i for i, l in enumerate(langues)}

    # u.normalise_data(dataset) # on le fait à l'intérieur du réseau
    return (corpus,)


@app.cell
def _(corpus):
    i=[] # j'ai ecrit ces lignes pour savoir quel sont les dites categories
    for elem in corpus:

        if elem.get('label') not in i: 
            i.append(elem.get('label'))
    print(i)
    return


@app.cell
def _(corpus):
    print( set([elem.get('label') for elem in corpus])) #essaie de recuperer les labrl par  list comprehension
    return


@app.cell
def _():
    classes={'culture':0, 'économie':1, 'science':2, 'international':3, 'société':4, 'sport':5, 'politique':6}
    return (classes,)


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
    train, test = train_test_split(corpus, train_size=0.8, random_state=42, )
    tv = keras.layers.TextVectorization(
            max_tokens=1000,
            #standardize=None,
            split="whitespace",
            ngrams=None, #(2,),
            output_mode="int",
            output_sequence_length=500
        )
    tv.adapt([line['text'] for line in train]) #list str qui correspond aux contenus des articles
    return test, train, tv


@app.cell
def _(classes, np, test, train, tv):
    X_train = tv([line['text'] for line in train])
    Y_train = np.array([classes[line['label']] for line in train])

    X_test = tv([line['text'] for line in test])
    Y_test = np.array([classes[line['label']] for line in test])
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(X_test, Y_train):
    print(X_test) # tensor 
    print(Y_train) #np array
    # pour mieux me rendre compte des donées que l'on exploite 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Création et entraînement d'un modèle
    (voir dans le fichier annexe pour les détails)
    """)
    return


@app.cell
def _(X_train, Y_train, classes, tv, u):
    model = u.create_model(tv.vocabulary_size(), len(classes))
    history= model.fit(X_train, Y_train, epochs=300, batch_size=100, validation_split=0.1)
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
def matrice_de_confusion(X_test, Y_test, classes, model, np, plt):
    from sklearn.metrics import ConfusionMatrixDisplay


    Y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        Y_test, # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1), 
        labels=range(len(classes)), 
        display_labels=classes
    )
    plt.show()
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _(history, plt):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'],label='Training Accuracy')
    plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
