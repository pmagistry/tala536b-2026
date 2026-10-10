import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _():
    # Source - https://stackoverflow.com/a/55406299
    # Posted by Tim K, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-10-09, License - CC BY-SA 4.0

    from keras.utils import plot_model
    from keras.layers import Activation, Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD

    import keras

    return Dense, SGD, Sequential, keras


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chargement de données (disponibles sur Moodle)
    """)
    return


@app.cell
def _(mo, np):
    def load_data():
        data = []
        with open("./corpus/corpus6.txt") as file:
            for line in file:
                label, text = line.strip().split(" ",1)
                n_th = text.count("th") #eng
                n_en = text.count("en")# allemand
                n_le = text.count("le")# fr
                n_os = text.count("os")#ajout pour detecter une troisieme langue es
                n_ze = text.count("ze") #nld
                n_io = text.count("io") #ita

                l = len(text)
                instance = {"label":label, "th": n_th / l, "en": n_en /l, "le": n_le/l,"os":n_os/l,"ze":n_ze/l, "io":n_io/l}
                data.append(instance)
        return data
    import random

    classes = {"deu": 0, "eng": 1, "spa": 2, "nld":3, "fra":4, "ita":5}   #pour les 6 langues

    all_data = [d for d in load_data() if d['label'] in classes] 
    random.seed(0) #rend ce melange reproductible
    random.shuffle(all_data) #melange
    data_lang = all_data[:500]
    #pourquoi melanger les données? prsq il se peut que les 500 premiers textes ne concerne qu'une langue donc on ne pourrait pas identifier 3 langues mais apprendrait a reconnaitre une langue par rapport a d'autres.
    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k]- mean) / std # forcer la disctribution des count loi normal centré en 0 avec un std de 1

    normalise_data(data_lang)

    # encodage des données pour les 6 features
    X6_lang=np.array([[d['th'], d['en'], d['le'],d['os'],d['io'],d['ze']] for d in data_lang]) 

    y6_lang = np.array([classes[d['label']] for d in data_lang])
    mo.md("essayez avec et sans normalisation !")
    return X6_lang, y6_lang


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Construction du modèle en Keras
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLP Keras
    """)
    return


@app.cell
def _(Dense, SGD, Sequential, X6_lang, keras, mo, y6_lang):
    # Construction du perceptron en Keras
    model6 = Sequential([
        keras.layers.Input(shape=(6,)),
        Dense(20,activation='tanh'),
        Dense(6, activation='sigmoid'),
    
    ])

    # Compilation du modèle
    model6.compile(
        optimizer=SGD(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model6.fit(X6_lang, y6_lang,
                        epochs=100, 
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2)

    # pseudo-Évaluation du modèle
    loss_keras6, acc_keras6 = model6.evaluate(X6_lang, y6_lang, verbose=0)

    # Affichage des poids appris
    weights6 = model6.layers[0].get_weights()[0]
    bias6 = model6.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras6:.2%}\n\nPoids pour 'en_freq' : {weights6[0][0]:.4f}\n\nPoids pour 'th_freq' : {weights6[1][0]:.4f}\n\nBiais : {bias6[0]:.4f}")
    return (model6,)


@app.cell
def _(model6):
    model6.summary()
    return


if __name__ == "__main__":
    app.run()
