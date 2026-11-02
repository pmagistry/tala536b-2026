import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Normalization
    from keras.optimizers import SGD
    import tools as tls
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    import matplotlib.pyplot as plt
    import tools as tls
    import tensorflow
    import numpy as np

    return mo, tls


@app.cell
def _(mo, tls):

    data = tls.load_json_data('../Data/corpus-classif.json')[:50]

    cat = set()
    cat.update([d['label'] for d in data])
    cat = list(cat)

    v, test, train = tls.vectorizer(data)
    texts = [e['text'] for e in train]

    vectorized = v(texts)

    model = tls.model_embedding(v, len(cat), 64, vectorized.shape[-1])
    y = tls.create_cat_onehot(train, cat, len(cat), mapping=True)



    model.fit(vectorized, y,
                epochs=10, 
                verbose=1, 
                shuffle=True)

    loss_keras, acc_keras = model.evaluate(v(texts), y, verbose=0)

    weights = model.layers[0].get_weights()[0]
    mo.md(f"Précision du modèle : {acc_keras:.2%}")
    return


if __name__ == "__main__":
    app.run()
