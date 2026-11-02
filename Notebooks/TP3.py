import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import os
    import json
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Normalization, Dropout
    from keras.optimizers import SGD
    from sklearn.model_selection import train_test_split
    import utils_tp3 as u
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    return (
        ConfusionMatrixDisplay,
        Dense,
        Dropout,
        Embedding,
        GlobalMaxPooling1D,
        SGD,
        Sequential,
        json,
        keras,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(json):
    with open('../data/corpus-classif.json', 'r') as f:
        json_data = json.load(f)
    return (json_data,)


@app.cell
def _(json_data):
    corpus = [{"label": d['catégorie'], "text": d['texte']} for d in json_data if d['texte'] != '\n' and d['catégorie'] != ""]
    categories = list(sorted({d['label'] for d in corpus}))
    code_categories = {l:i for i, l in enumerate(categories)}
    return categories, code_categories, corpus


@app.cell
def _(corpus, keras, train_test_split):
    train, test = train_test_split(corpus, train_size=0.8, random_state=42)
    tv = keras.layers.TextVectorization(
        max_tokens=2000,
        standardize='lower_and_strip_punctuation',
        split="whitespace",
        ngrams=None,
        output_mode="int",
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
def _(Dense, Dropout, Embedding, GlobalMaxPooling1D, SGD, Sequential):
    def model_emb(nb_input, nb_classes):
        model = Sequential([
            Embedding(input_dim=nb_input, output_dim=128),
            GlobalMaxPooling1D(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(nb_classes, activation='softmax')
        ])
        model.compile(
            optimizer=SGD(learning_rate=0.1, momentum=0.2),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    return (model_emb,)


@app.cell
def _(X_train, Y_train, categories, model_emb, tv):
    model = model_emb(tv.vocabulary_size(), len(categories))
    history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_split=0.10)
    return (model,)


@app.cell
def _(X_test, Y_test, model):
    model.evaluate(X_test, Y_test)
    return


@app.cell
def _(ConfusionMatrixDisplay, X_test, Y_test, categories, model, np, plt):
    Y_pred = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        Y_test, # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1), 
        labels=range(len(categories)), 
        display_labels=categories
    )
    plt.show()
    return


@app.cell
def _(tv):
    tv.get_vocabulary()
    return


@app.cell
def _(model):
    model.summary()
    return


if __name__ == "__main__":
    app.run()
