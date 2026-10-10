import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import polars as pl
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    return np, os, train_test_split


@app.cell
def _(np, train_test_split):
    # chargement
    bigrams = [
        "th",
        "en",
        "wh",
        "qu",
        "ux",
        "mp",
        "ns",
        "ee",
        "ze",
        "ci",
        "oo",
        "pp",
        "mm",
        "nn",
    ]

    def load_data():
        data = []
        with open("./corpora/corpus_6.txt") as file:
            for line in file:
                label, text = line.strip().split(" ", 1)
                instance = {
                    "label": label,
                }
                for b in bigrams:
                    instance[b] = text.count(b) / len(text)
                data.append(instance)
        return data

    data_lang = load_data()[:200]

    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != "label":
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k] - mean) / std

    normalise_data(data_lang)

    # test du nouveau réseau de neurones pour 2 langues
    # data_lang = list(filter(lambda d: d["label"] in ["eng", "deu"], data_lang))
    # classes = ["eng", "deu"]

    print(*data_lang[:40], sep="\n")

    # encodage des données
    X_lang = np.array([[d[b] for b in bigrams] for d in data_lang])

    # distribution de probabilités entre les classes
    classes = ["eng", "deu", "nld", "fra", "ita", "spa"]
    n_class = len(classes)

    # 1 pour la bonne langue, 0 pour les autres
    y_lang = np.array(
        [
            [
                1 if lang == classes.index(d["label"]) else 0
                for lang, i in enumerate(classes)
            ]
            for d in data_lang
        ]
    )

    print(*X_lang[:10], sep="\n")
    print(classes)
    print(*y_lang[:10], sep="\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_lang, y_lang, test_size=0.2
    )
    return X_test, X_train, bigrams, classes, n_class, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en Keras
    """)
    return


@app.cell
def _(os):
    # imports keras
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, ReLU
    from keras.optimizers import SGD

    return Dense, SGD, Sequential, keras


@app.cell
def _(
    Dense,
    SGD,
    Sequential,
    X_test,
    X_train,
    bigrams,
    keras,
    mo,
    n_class,
    y_test,
    y_train,
):
    # Construction du réseau neuronal en Keras
    model = Sequential(
        [
            keras.layers.Input(shape=(len(bigrams),)),
            Dense(20, activation="linear"),
            Dense(10, activation="relu"),
            Dense(n_class, activation="softmax"),
        ]
    )

    # Compilation du modèle
    model.compile(
        optimizer=SGD(learning_rate=0.2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Entraînement du modèle
    model.fit(
        X_train,
        y_train,
        epochs=50,
        verbose=False,
        shuffle=True,
        validation_split=0.2,
    )

    # Évaluation du modèle
    loss_keras, acc_keras = model.evaluate(X_test, y_test, verbose=0)

    # Affichage des poids appris
    mo.md(f"""
            Coût final : {loss_keras:.2}
            Précision du modèle : {acc_keras:.2%}
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction et entraînement du modèle en pytorch
    """)
    return


@app.cell
def _():
    # imports pytorch
    import torch
    import torch.nn as nn

    return nn, torch


@app.cell
def _(X_test, X_train, bigrams, n_class, nn, np, torch, y_test, y_train):
    # Conversion des matrices numpy en tensors pytorch
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(
        y_train.astype(np.float32).reshape(y_train.shape[0], n_class)
    )
    X_test_t = torch.from_numpy(X_test.astype(np.float32))
    y_test_t = torch.from_numpy(
        y_test.astype(np.float32).reshape(y_test.shape[0], n_class)
    )

    # Construction du réseau neuronal en pytorch
    model_t = nn.Sequential(
        nn.Linear(len(bigrams), 20),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, n_class),
    )
    learning_rate_t = 0.2
    optimizer_t = torch.optim.SGD(model_t.parameters(), lr=learning_rate_t)

    # Fonction coût
    loss_fn_t = nn.CrossEntropyLoss()
    n_epochs = 50

    for i in range(n_epochs):
        # prédiction
        yhat_t = model_t(X_train_t)

        # mise à 0 des gradients
        optimizer_t.zero_grad()
        # coût
        loss_t = loss_fn_t(yhat_t, y_train_t)
        if i == n_epochs - 1:
            print(f"Coût final : {loss_t}")

        # backpropagation
        loss_t.backward()

        # mise à jour des poids
        optimizer_t.step()
    return X_test_t, model_t, y_test_t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Évalutation du modèle pytorch
    """)
    return


@app.cell
def _(X_test_t, model_t, torch, y_test_t):
    # modele pytorch en mode evaluation
    model_t.eval()

    n_correct = 0
    n_examples = len(X_test_t)

    # desactive le calcul de gradient le temps de faire seulement de l'inference
    with torch.no_grad():
        _, predictions = torch.max(model_t(X_test_t), 1)
        y_test_index = torch.argmax(y_test_t, 1)
        predictions = predictions.cpu()
        y_test_index = y_test_index.cpu()
        n_correct = (predictions == y_test_index).sum().item()

    accuracy = 100 * n_correct / n_examples
    print(f"Accuracy du modèle pytorch : {accuracy}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test du modèle pytorch
    """)
    return


@app.cell
def _(bigrams, classes, model_t, torch):
    def predict_lang(sentence):
        sentence_tensor = torch.tensor(
            [float(sentence.count(b)) for b in bigrams]
        )

        return classes[torch.argmax(model_t(sentence_tensor))]

    # test_sentence = "Ernest Boulanger had studied at the Paris Conservatoire and, in 1835 at the age of 20, won the coveted Prix de Rome for composition. He wrote comic operas and incidental music for plays but was most widely known for his choral music. He achieved distinction as a director of choral groups, teacher of voice, and a member of choral competition juries. After years of rejection, in 1872 he was appointed to the Paris Conservatoire as professor of singing."
    test_sentence = "m Jahr 1914 komponierte sie „Drei Stücke“ für Violoncello und Klavier, ein impressionistisches Werk mit drei Teilen jeweils eigenen Charakters. Der Pianist Raoul Pugno (1852–1914) setzte sich für Nadia Boulanger ein und führte unter ihrer Leitung ihre Rhapsodie variée für Klavier und Orchester auf. Auch komponierte er mit ihr gemeinsam eine Reihe von Werken wie den Liederzyklus der Heures claires („Helle Stunden“). Nach seinem Tod widmete Nadia Boulanger sich stärker der Musikpädagogik, Orchesterleitung und der Verbreitung des Werks ihrer Schwester Lili Boulanger. Ab 1921 unterrichtete sie an der École normale de musique de Paris und am neu gegründeten Conservatoire Américain in Fontainebleau. Im selben Jahr reiste sie erstmals in die USA, wo sie fortan regelmäßig Meisterkurse gab. Sie wurde eine der berühmtesten Kompositionslehrerinnen des 20. Jahrhunderts."

    print(f"Langue du texte : {predict_lang(test_sentence)}")
    return (predict_lang,)


@app.cell
def _():
    lang_map = {
        "eng": "Anglais",
        "deu": "Allemand",
        "nld": "Néerlandais",
        "fra": "Français",
        "ita": "Italien",
        "spa": "Espagnol",
    }
    return (lang_map,)


@app.cell
def _(mo):
    form = mo.ui.text_area(placeholder="...").form()
    return (form,)


@app.cell
def _(form, lang_map, mo, predict_lang):
    mo.vstack(
        [form, mo.md(f"Langue : {lang_map[predict_lang(form.value or '')]}")]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus : Test d'un modèle keras avec TextVectorizer
    """)
    return


@app.cell
def _():
    # from tensorflow.keras.layers import TextVectorization

    # vectorize_layer = TextVectorization(
    # max_tokens=5000,
    # output_mode='int',
    # output_sequence_length=10)

    # vectorize_layer.adapt(X_lang)

    # # Construction du réseau neuronal en Keras
    # model_tv = Sequential(
    #     [
    #         keras.layers.Input(shape=(10,)),
    #         Dense(20, activation="linear"),
    #         Dense(10, activation="relu"),
    #         Dense(n_class, activation="softmax"),
    #     ]
    # )

    # # Compilation du modèle
    # model_tv.compile(
    #     optimizer=SGD(learning_rate=0.2),
    #     loss="categorical_crossentropy",
    #     metrics=["accuracy"],
    # )

    # # Entraînement du modèle
    # model_tv.fit(
    #     X_train,
    #     y_train,
    #     epochs=50,
    #     verbose=False,
    #     shuffle=True,
    #     validation_split=0.2,
    # )

    # # Évaluation du modèle
    # loss_tv, acc_tv = model_tv.evaluate(X_test, y_test, verbose=0)

    # # Affichage des poids appris
    # mo.md(f"""
    #         Coût final : {loss_tv:.2}
    #         Précision du modèle : {acc_tv:.2%}
    #     """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
