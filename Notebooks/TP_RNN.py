import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import utils_RNN

    return Path, mo


@app.cell
def _(Path):
    DATA = Path("Data")
    fichiers = {
        "train": DATA / "fa_perdt-ud-train.conllu",
        "dev": DATA / "fa_perdt-ud-dev.conllu",
        "test": DATA / "fa_perdt-ud-test.conllu",
    }
    return (fichiers,)


@app.cell
def _(fichiers, pos_utils):
    X_train, y_train = pos_utils.read_conllu(fichiers["train"])
    X_dev, y_dev = pos_utils.read_conllu(fichiers["dev"])
    X_test, y_test = pos_utils.read_conllu(fichiers["test"])
    return X_dev, X_test, X_train, y_train


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
def _(X_train, y_train):
    list(zip(X_train[0], y_train[0]))
    return


@app.cell
def _(pos_utils, y_train):
    pos_utils.compter_etiquettes(y_train).most_common()
    return


if __name__ == "__main__":
    app.run()
