import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp3 as u
    from matplotlib import pyplot as plt

    return keras, np, plt, train_test_split, u


@app.cell
def _(u):
    import importlib
    importlib.reload(u)
    return


@app.cell
def _(u):
    corpus = u.load_text("./corpus-classif.json", limit=500)
    categories = list(sorted({ d['label'] for d in corpus }))
    code_categories = {c:i for i, c in enumerate(categories)}
    return categories, code_categories, corpus


@app.cell
def _():
    max_tokens = 1000
    split = "whitespace"
    ngrams = None
    output_mode = "int"
    sequence_length = 1000
    return max_tokens, ngrams, output_mode, sequence_length, split


@app.cell
def _(
    corpus,
    keras,
    max_tokens,
    ngrams,
    output_mode,
    sequence_length,
    split,
    train_test_split,
):
    train, test = train_test_split(corpus, train_size=0.8, random_state=42)
    tv = keras.layers.TextVectorization(
            max_tokens=max_tokens,
            standardize=None,
            split=split,
            ngrams=ngrams,
            output_mode=output_mode,
            output_sequence_length=sequence_length,
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
def _(X_train, Y_train, categories, sequence_length, tv, u):
    model = u.model_embedding(
        nb_input=sequence_length,
        nb_classes=len(categories),
        vocab_size=tv.vocabulary_size()
    )
    #model.layers[0].adapt(X_train)
    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    return history, model


@app.cell
def _(X_test, Y_test, model):
    model.evaluate(X_test, Y_test)
    return


@app.cell
def _(history, plt):
    plt.figure(figsize=(10,6))
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
