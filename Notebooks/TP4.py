import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    import regex
    import os
    import json
    import random
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    from sklearn.model_selection import train_test_split
    import utils_tp4 as u

    return (u,)


@app.cell
def _(u):
    corpus = u.load_conllu("../data/fr_parisstories-ud-dev.conllu")
    return


if __name__ == "__main__":
    app.run()
