import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Disclaimer
    j'ai adapté sans trop de problèmes les fonctions de prétraitement,
    j'en ai réjouté pour créer un vocabulaire et le word2Index,
    j'avais à peu près quelque chose de convaincant pour la vectorisation mais j'ai pas eu l'intuition
    de faire par batch pour résoudre le problème de padding des phrases que j'avais
    pour le reste, c'est essentiellement généré par GLM 5.3 Flash (via mistral)
    j'ai essayé de comprendre autant que possible mais ça reste cryptique
    je comprends les entrées et les sorties de chaque fonction mais tout ce qu'il y a entre me reste opaque
    mais voilà, disclaimer pour dire que je n'ai clairement pas compris tout ce qui est écrit ici (dans le fichier utils, ici ça va c'est ok)
    """)
    return


@app.cell
def _():
    import marimo as mo
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import utils_tp3_torch as u

    return ConfusionMatrixDisplay, mo, plt, train_test_split, u


@app.cell
def _(u):
    corpus, classes, code_classes = u.load_data_json("../data/corpus-classif.json")
    return classes, code_classes, corpus


@app.cell
def _(code_classes, corpus, train_test_split, u):
    train, test = train_test_split(corpus, train_size=0.8, random_state=42)

    vocab = u.create_vocab(train)
    word2Index = u.create_word2Index(vocab)

    Y_train = [code_classes[s["label"]] for s in train]
    Y_test = [code_classes[s["label"]] for s in test]
    return Y_test, Y_train, test, train, vocab, word2Index


@app.cell
def _(Y_test, Y_train, classes, test, train, u, vocab, word2Index):
    model, lr = u.model_embedding(len(vocab), embed_dim=50, nb_classes=len(classes))
    model = u._fit(model, train, Y_train, word2Index, lr=lr)
    u.evaluate_model(model, test, Y_test, word2Index)
    return (model,)


@app.cell
def matrice_de_confusion(
    ConfusionMatrixDisplay,
    Y_test,
    classes,
    model,
    plt,
    test,
    u,
    word2Index,
):
    import numpy as np

    Y_pred = u.predict_model(model, test, word2Index)

    ConfusionMatrixDisplay.from_predictions(
        Y_test,  # en "sparse", Y est déjà bien encodé
        Y_pred,
        labels=range(len(classes)),
        display_labels=classes
    )
    plt.show()
    return


@app.cell
def _(Y_test, Y_train, classes, test, train, u, weights, word2Index):
    weights
    pretrained_model, lr = u.model_embedding_pretrained(weights, embed_dim=50, nb_classes=len(classes))
    pretrained_model = u._fit(pretrained_model, train, Y_train, word2Index, lr=lr)
    u.evaluate_model(pretrained_model, test, Y_test, word2Index)
    return


if __name__ == "__main__":
    app.run()
