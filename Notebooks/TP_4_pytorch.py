import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import tools as tls
    import utils_pytorch as ut
    from sklearn.model_selection import train_test_split

    return tls, ut


@app.cell
def _(tls):
    data = tls.load_json_data('./Data/corpus-classif.json')[:200]

    cat = set()
    cat.update([d['label'] for d in data])
    cat = list(cat)

    v, train, test = tls.vectorizer(data)

    vectorized_train = v([e['text'] for e in train])
    vectrorized_test = v([e['text'] for e in test])

    y = tls.create_cat_onehot(train, cat, len(cat), mapping=True)
    return cat, vectorized_train, y


@app.cell
def _(cat, ut, vectorized_train, y):
    model, optimizer, loss_function = ut.model_embeddings_pt(
        vectorized_train.shape[-1],
        128,
        64,
        len(cat)
    )


    accuracy = ut.train_model(model, vectorized_train, y, 100, optimizer, loss_function)

    print(accuracy)
    return


if __name__ == "__main__":
    app.run()
