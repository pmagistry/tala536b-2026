import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    from utils_tp2 import load_data, get_vocab, vectorise, normalise_data

    return (
        ConfusionMatrixDisplay,
        Dense,
        SGD,
        Sequential,
        classification_report,
        get_vocab,
        keras,
        load_data,
        mo,
        normalise_data,
        np,
        train_test_split,
        vectorise,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP2 : classification en 6 langues

    ## Chargement des donnees
    """)
    return


@app.cell
def _(load_data, np):
    texts, labels = load_data("./corpus6.txt")
    languages = sorted(set(labels))
    y = np.array([languages.index(l) for l in labels])
    (len(texts), languages)
    return languages, texts, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Découpage train / dev / test

    80/10/10
    """)
    return


@app.cell
def _(texts, train_test_split, y):
    texts_train, texts_rest, y_train, y_rest = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=0)
    texts_dev, texts_test, y_dev, y_test = train_test_split(
        texts_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)
    (len(texts_train), len(texts_dev), len(texts_test))
    return texts_dev, texts_test, texts_train, y_dev, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Traits : fréquences de bigrammes de lettres
    """)
    return


@app.cell
def _(normalise_data, np, texts_dev, texts_test, texts_train, vectorise):
    def prep(vocab):
        X_train = vectorise(texts_train, vocab)
        X_dev = vectorise(texts_dev, vocab)
        X_test = vectorise(texts_test, vocab)

        X = np.concatenate([X_train, X_dev, X_test])
        normalise_data(X)
        n1 = len(X_train)
        n2 = n1 + len(X_dev)
        X_train, X_dev, X_test = X[:n1], X[n1:n2], X[n2:]
        return X_train, X_dev, X_test

    return (prep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modèle

    - entrée : un trait par bigramme du vocab (3, 20 ou 100)
    - une couche cachée de 32 neurones (tanh)
    - sortie : 6 neurones avec softmax (une probabilité par langue)
    - loss : `sparse_categorical_crossentropy` car les étiquettes sont des entiers 0..5
      (avec du one-hot ce serait `categorical_crossentropy`)
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, keras, languages, y_dev, y_train):
    def train_model(X_train, X_dev):
        model = Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            Dense(32, activation="tanh"),
            Dense(len(languages), activation="softmax")
        ])

        model.compile(
            optimizer=SGD(learning_rate=0.05),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(X_train, y_train,
                            epochs=5,
                            batch_size=64,
                            verbose=1,
                            validation_data=(X_dev, y_dev))
        return model, history

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Affichage de l'evaluation
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    classification_report,
    languages,
    mo,
    texts_test,
    y_test,
):
    def evaluate_model(model, X_test):
        loss_test, acc_test = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

        errors = [(languages[y_test[i]], languages[y_pred[i]], texts_test[i])
                  for i in range(len(y_test)) if y_test[i] != y_pred[i]]

        return mo.vstack([
            mo.md(f"""
    Accuracy sur le test : {acc_test:.2%}

    ```
    {classification_report(y_test, y_pred, target_names=languages, digits=3)}
    ```
    """),
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=languages).figure_,
            errors[:10],
        ])

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 Bigrames (th, en, le)
    """)
    return


@app.cell
def _(evaluate_model, prep, train_model):
    vocab_3 = ['th', 'en', 'le']
    X_train_3, X_dev_3, X_test_3 = prep(vocab_3)
    model_3, hist_3 = train_model(X_train_3, X_dev_3)
    evaluate_model(model_3, X_test_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 20 bigrammes les plus communs
    """)
    return


@app.cell
def _(evaluate_model, get_vocab, prep, texts_train, train_model):
    vocab_20 = get_vocab(texts_train, k=20)
    X_train_20, X_dev_20, X_test_20 = prep(vocab_20)
    model_20, hist_20 = train_model(X_train_20, X_dev_20)
    evaluate_model(model_20, X_test_20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 100 bigrammes les plus communs
    """)
    return


@app.cell
def _(evaluate_model, get_vocab, prep, texts_train, train_model):
    vocab_100 = get_vocab(texts_train, k=100)
    X_train_100, X_dev_100, X_test_100 = prep(vocab_100)
    model_100, hist_100 = train_model(X_train_100, X_dev_100)
    evaluate_model(model_100, X_test_100)
    return


if __name__ == "__main__":
    app.run()
