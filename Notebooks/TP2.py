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

    return mo, tls


@app.cell
def _(mo, tls):
    # perceptron multicouche pour 6 langues 

    # pour choisir les bigrammes qui vous nous permettre de différentier les langues entre elles, j'ai cherché quels étaient les bigrammes les plus fréquents dans chaque langues et j'ai comparé ces résulats avec ce qui 'était observables dans le corpus, afin de choisir des candidats ayant un plus haut potentiel de discrimination entre les langues

    classes = ['eng', 'fra', 'deu', 'nld', 'spa', 'ita']
    bigrammes = ['th', 'le', 'en', 'aa', 'ci', 'zi']
    mapping = {l : i for i,l in enumerate(classes)}
    reverse_map = {v : k for k,v in mapping.items()}

    data_lang_6 = tls.load_data('../Data/corpus6.txt', bigrammes, mapping)[:200]

    tls.normalise_data(data_lang_6)

    x, y = tls.format_data(data_lang_6, classes, 'onehot')


    model_6 = tls.model_multiclass_categorical_bigramms(6, 64, 6, norm=True)

    model_6.fit(x, y,
                epochs=10, 
                verbose=1, 
                shuffle=True, 
                validation_split=0.2)

    loss_keras_6, acc_keras_6 = model_6.evaluate(x, y, verbose=0)

    weights_6 = model_6.layers[0].get_weights()[0]
    bias_6 = model_6.layers[0].get_weights()[1]
    mo.md(f"Précision du modèle : {acc_keras_6:.2%}\n\nPoids pour 'le_freq' : {weights_6[0][0]:.4f}\n\nPoids pour 'th_freq' : {weights_6[1][0]:.4f}\n\nPoids pour 'en_freq' : {weights_6[2][0]:.4f}\n\nPoids pour 'aa_freq' : {weights_6[3][0]:.4f}\n\nPoids pour 'ci_freq' : {weights_6[4][0]:.4f}\n\nPoids pour 'zi_freq' : {weights_6[5][0]:.4f}\n\nBiais : {bias_6[0]:.4f}")
    return bigrammes, classes, mapping, model_6, reverse_map


@app.cell
def _(bigrammes, classes, mapping, model_6, reverse_map, tls):
    # évaluation du modèle 

    """
    dans l'exercice du dessus, on utilise les 200 premières lignes du corpus pour procéder à l'entrainement du modèle

    pour le tester, on va donc prendre d'autres données du même corpus
    """

    sample = tls.load_data('../Data/corpus6.txt', bigrammes, mapping)[1000:1100]

    real_answers = [d['label'] for d in sample]

    tls.normalise_data(sample)

    data = tls.format_data(sample, classes, 'predict')

    predictions = model_6.predict(data)

    match = 0

    for i, pred in enumerate(predictions) : 
        print(f"Langue Prédite : {tls.format_prediction_onehot(pred, reverse_map)} / Bonne Catégorie : {reverse_map[real_answers[i]]}")
        if tls.format_prediction_onehot(pred, reverse_map) == reverse_map[real_answers[i]]:
            match += 1
    print(f"{match / len(predictions) * 100}%")
    return


if __name__ == "__main__":
    app.run()
