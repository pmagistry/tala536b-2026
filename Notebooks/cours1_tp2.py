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
def _(np):
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
                n_as = text.count("as")
                n_re = text.count("re") 
                n_zu = text.count("zu")
                n_to = text.count("to")
                n_ve = text.count("ve")
                n_iu = text.count("iu")
                #j'ai choisi au debut  d'avoir 1 feature par langue mais je sais que je peux avoir plus de features que de langues a identifié ( car les entrées et les sorties du mlp sont décorrélés) maisfinalement ce n'etait pas suffisant pour le modele donc j'ai ajoutés des features 
            
                l = len(text)
                instance = {"label":label, "th": n_th / l, "en": n_en /l, "le": n_le/l,
                           "os":n_os/l,"ze":n_ze/l, "io":n_io/l,
                           "as":n_as/l, "re":n_re/l, "zu":n_zu/l, 
                           "to":n_to/l, "ve":n_ve/l, "iu":n_iu/l, }
                data.append(instance)
        return data
    import random
    classes = {"deu": 0, "eng": 1, "spa": 2, "nld":3, "fra":4, "ita":5}   #pour les 6 langues

    all_data = [d for d in load_data() if d['label'] in classes] 
    random.seed(0) #rend ce melange reproductible
    random.shuffle(all_data) #melange
    data_lang = all_data[:4000]
    #pourquoi melanger les données? prsq il se peut que les 4000 premiers textes ne concerne qu'une langue donc on ne pourrait pas identifier 3 langues mais apprendrait a reconnaitre une langue par rapport a d'autres.
    # normalisation
    def normalise_data(dataset):
        for k in dataset[0].keys():
            if k != 'label':
                mean = np.mean([d[k] for d in dataset])
                std = np.std([d[k] for d in dataset])
                for d in dataset:
                    d[k] = (d[k]- mean) / std # forcer la disctribution des count loi normal centré en 0 avec un std de 1

    normalise_data(data_lang)

    # encodage des données pour les 12 features
    features = ["th", "en", "le", "os", "ze", "io", "as", "re", "zu", "to", "ve","iu"]
    X6_lang = np.array([[d[f] for f in features] for d in data_lang])

    y6_lang = np.array([classes[d['label']] for d in data_lang])
    #decoupage propore des donées entrainement et de test 
    from sklearn.model_selection import train_test_split
    X6_train, X6_test, y6_train, y6_test = train_test_split(
        X6_lang, y6_lang,
        test_size=0.2,        #20% mis de côté pour le test final
        random_state=0,       #pour un découpage reproductible
        stratify=y6_lang,     #les mêmes proportions de chaque langue dans train et test
    )
    return X6_test, X6_train, classes, features, y6_test, y6_train


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
def _(
    Dense,
    SGD,
    Sequential,
    X6_test,
    X6_train,
    features,
    keras,
    mo,
    y6_test,
    y6_train,
):
    keras.utils.set_random_seed(812) # Source - https://keras.io/examples/keras_recipes/reproducibility_recipes/
    # Construction du perceptron en Keras
    model6 = Sequential([
        keras.layers.Input(shape=(len(features),)), # modifié le hard code par la longueur de ma variable feature comme ca si j'en ajoute d'autres je n'ai pas besoin de me rappeler qu'il faut que je modifie cette partie aussi.
        Dense(16,activation='relu'), #peut etre modifie tanh par relu 
        Dense(6, activation='softmax'),
    
    ])

    # Compilation du modèle
    model6.compile(
        optimizer=SGD(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement du modèle
    model6.fit(X6_train, y6_train,  #au lieu de X6_lang et y6_lang 
                        epochs=100, 
                        verbose=1, 
                        shuffle=True, 
                        validation_split=0.2)

    # Évaluation du test
    loss_keras6, acc_keras6 = model6.evaluate(X6_test, y6_test, verbose=0) #corpus dedié au test pas encore vue par le modele

    mo.md(f"Précision validation : {model6.history.history['val_accuracy'][-1]:.2%}\n\n"
          f"Précision test : {acc_keras6:.2%}")  
    #pour garder que la dernier val accuracy car on a ajouté une couche caché # source -- https://stackoverflow.com/    questions/51731207/python-neural-network-typeerror-history-object-is-not-subscriptable
    return (model6,)


@app.cell
def _(model6):
    model6.summary()
    return


@app.cell
def _(X6_test, classes, model6, y6_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


    predictions = model6.predict(X6_test).argmax(axis=1)

    cm = confusion_matrix(y6_test, predictions, labels=list(classes.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(classes.keys()))
    disp.plot()

    plt.show()
    return


if __name__ == "__main__":
    app.run()
