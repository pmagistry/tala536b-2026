import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import utils_RNN

    return Path, mo, utils_RNN


@app.cell
def _(Path):
    DATA = Path("Data")
    files = {
        "train": DATA / "fa_perdt-ud-train.conllu",
        "dev": DATA / "fa_perdt-ud-dev.conllu",
        "test": DATA / "fa_perdt-ud-test.conllu",
    }
    return (files,)


@app.cell
def _(files, utils_RNN):
    X_train, Y_train = utils_RNN.read_conllu(files["train"])
    X_dev, Y_dev = utils_RNN.read_conllu(files["dev"])
    X_test, Y_test = utils_RNN.read_conllu(files["test"])
    return X_dev, X_test, X_train, Y_dev, Y_test, Y_train


@app.cell
def _(X_dev, X_test, X_train, mo):
    mo.md(f"""
    | partie | phrases | mots |
    |---|---|---|
    | train | {len(X_train)} | {sum(map(len, X_train))} |
    | dev | {len(X_dev)} | {sum(map(len, X_dev))} |
    | test | {len(X_test)} | {sum(map(len, X_test))} |
    """)
    return


@app.cell
def _(X_train, Y_train):
    list(zip(X_train[0], Y_train[0]))
    return


@app.cell
def _(Y_train, utils_RNN):
    utils_RNN.count_tags(Y_train).most_common()
    return


@app.cell
def _(X_dev, X_test, X_train, Y_dev, Y_test, Y_train, utils_RNN):
    word_to_num, tag_to_num = utils_RNN.build_vocabs(X_train, Y_train)

    X_train_enc = utils_RNN.encode(X_train, word_to_num)
    X_dev_enc = utils_RNN.encode(X_dev, word_to_num)
    X_test_enc = utils_RNN.encode(X_test, word_to_num)
    Y_train_enc = utils_RNN.encode(Y_train, tag_to_num)
    Y_dev_enc = utils_RNN.encode(Y_dev, tag_to_num)
    Y_test_enc = utils_RNN.encode(Y_test, tag_to_num)
    return (
        X_dev_enc,
        X_test_enc,
        X_train_enc,
        Y_dev_enc,
        Y_test_enc,
        Y_train_enc,
        tag_to_num,
        word_to_num,
    )


@app.cell
def _(
    X_dev_enc,
    X_test_enc,
    X_train,
    X_train_enc,
    Y_dev_enc,
    Y_test_enc,
    Y_train_enc,
    utils_RNN,
):
    max_len = max(map(len, X_train))

    X_train_pad = utils_RNN.padding(X_train_enc, max_len)
    X_dev_pad = utils_RNN.padding(X_dev_enc, max_len)
    X_test_pad = utils_RNN.padding(X_test_enc, max_len)
    Y_train_pad = utils_RNN.padding(Y_train_enc, max_len)
    Y_dev_pad = utils_RNN.padding(Y_dev_enc, max_len)
    Y_test_pad = utils_RNN.padding(Y_test_enc, max_len)
    return X_dev_pad, X_train_pad, Y_dev_pad, Y_train_pad, max_len


@app.cell
def _(max_len, tag_to_num, utils_RNN, word_to_num):
    model = utils_RNN.build_model(len(word_to_num), len(tag_to_num), max_len)
    model.summary()
    return (model,)


@app.cell
def _(X_dev_pad, X_train_pad, Y_dev_pad, Y_train_pad, model):
    history = model.fit(
        X_train_pad, Y_train_pad,
        validation_data=(X_dev_pad, Y_dev_pad),
        epochs=5,
        batch_size=32,
    )
    return


if __name__ == "__main__":
    app.run()
