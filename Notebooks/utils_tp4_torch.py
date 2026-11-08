from pathlib import Path
import json
import regex
import numpy as np
import polars as pl
import os
import torch

from collections import Counter

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

def lower_and_filter_names(txt: str) -> str:
    """
    Fonction de prétraitement du texte
    """
    txt = regex.sub("\\p{P}", " ", txt) # enlever la ponctuation
    txt = " ".join(txt.lower().split())
    for name in ["figaro", "libération", "france-info", "bfmtv", "bfm"]:
        txt = txt.replace(name, "") # supprime quelques indices trop forts
    return txt


def load_data(corpus_file="./corpus-classif.json", target="catégorie") -> dict:
    """ 
    Chargement du corpus
    Retourne le corpus, les classes et un dict associant les classes à leur code.
    """
    with open(corpus_file) as _f:
        data = json.load(_f)
    corpus = [{'label': _i[target], 'text': lower_and_filter_names(_i['texte'])} for _i in data if len(_i['texte'].strip().split()) > 20 and _i[target] != '']
    classes = list(sorted({ d['label'] for d in corpus }))
    code_classes = {l:i for i, l in enumerate(classes)}
    return corpus, classes, code_classes

def build_vocabulary(text, vocab_size = 5000):
    compteur = Counter()
    for t in text:
        compteur.update(t.split())
    frq = compteur.most_common(vocab_size - 2)
    vocab = {}
    for numero, (mot, nombre) in enumerate(frq, start = 2):
        vocab[mot] = numero
    return vocab

def create_textvectorizer(corpus, vocab_size=1000):
    text = [line['text'] for line in corpus]
    build_vocabulary(text, vocab_size)

def stratified_split(corpus, seed=42):
    """
        retourne trois tuples: train, valid, test
        chaque sous-corpus est un tuple (X,Y)
    """
    texts = [r["text"] for r in corpus]
    y = [r["label"] for r in corpus]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        texts, y, test_size=0.30, stratify=y, random_state=seed
    )
    x_dev, x_te, y_dev, y_te = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return (x_tr, y_tr), (x_dev, y_dev), (x_te, y_te)

def encode_text(text, vocab, max_len = 30):
    mots = text.split()
    num = []
    for mot in mots:
        if mot in vocab:
            num.append(vocab[mot])
        else:
            num.append(1)
    num = num[:max_len]
    while len(num) < max_len:
        num.append(0)
    return num

def encode(subcorpus, tv, encodeur):
    texts = subcorpus[0]  # liste des textes
    labels = subcorpus[1]  # liste des catégories (en mots)
    X = []
    for text in texts:
        X.append(encode_text(text, tv, 30))
    Y = []
    for label in labels:
        Y.append(encodeur[label])
    return torch.tensor(X), torch.tensor(Y)

def create_model_seq(nb_input, vocab: list[str], emb_size=200, nb_classes=6, dropout=0.5, optim="sgd"):
    pass

def create_model_func(nb_input, vocab: list[str], emb_size=200, nb_classes=6, dropout=0.5, optim="sgd"):
   pass


def plot_history(history):
    """
        fonction d'affichage de l'évolution de l'exactitude
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.gca()
    return fig




def plot_confusion(model, X_test, Y_test, labels):
    Y_pred = model.predict(X_test)

    m = ConfusionMatrixDisplay.from_predictions(
        Y_test, # en "sparse", Y est déjà bien encodé
        np.argmax(Y_pred, axis=1), 
        labels=range(len(labels)), 
        display_labels=labels
    )
    plt.show()
    return m


def plot_embeddings(model, tv, layer_idx=1):
    """
        affichage interactif de plongements lexicaux
        va chercher les plongements à la `layer_idx`-eme couche du modèle
        (on pourrait utiliser les argument name= pour la retrouver plus facilement)
    """
    import plotly.express as px
    from umap import UMAP
    _N_min=100
    _N_max=1000
    embedding_layer = model.layers[layer_idx]
    embeddings = embedding_layer.get_weights()[0]

    vocabulary = tv.get_vocabulary()

    umap = UMAP(n_components=2, n_neighbors=10, min_dist=0.2, metric='cosine', low_memory=False)
    embeddings_2d = umap.fit_transform(embeddings)

    df_embeddings = pl.DataFrame({
        'x': embeddings_2d[_N_min:_N_max, 0],
        'y': embeddings_2d[_N_min:_N_max, 1],
        'word': vocabulary[_N_min:_N_max]
    })

    fig = px.scatter(
        df_embeddings,
        x='x',
        y='y',
        text='word',
        title='UMAP Visualization of Embeddings from the First Layer (Interactive)',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        hover_name='word'
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.5),
        textposition='top center'
    )

    fig.update_layout(
        height=800,
        width=800,
        showlegend=False,
        hovermode='closest'
    )

    return fig




# keras.layers.Embedding(voc_size, 
#                                emb_size,
#                                embeddings_initializer=u.PretrainedInitializer(
#                                    "frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", 
#                                    vocab
#                                ),
#                                # embeddings_regularizer=keras.regularizers.L1L2(),
#                                # embeddings_constraint=keras.constraints.MinMaxNorm(),
#                                trainable=True
#
#                               )



class PretrainedInitializer(keras.initializers.Initializer):
    """
        classe qui permet d'initialiser des Embeddings avec un fichier
        binaire au format word2vec
        (voir ci-dessus pour son utilisation)
    """

    def __init__(self, data_file: str, vocab: list[str]):
        self.data_file = data_file
        self.vocab = vocab
        self.w2v = KeyedVectors.load_word2vec_format(data_file, binary=True, unicode_errors='ignore')


    def __call__(self, shape, dtype=None):
        weights = keras.initializers.GlorotUniform()(shape, dtype)
        for i, w in enumerate(self.vocab):
            if w in self.w2v:
                weights[i] = torch.tensor(self.w2v[w])
        return weights


    #def get_config(self):  # To support serialization
    #  return {'data_file': self.data_file, 'vocab': self.vocab}


def scheduler(epoch, lr):
    if epoch < 150:
        return (0.0001 * (epoch+1)/ 50)
    elif epoch < 150:
        return 0.0001
    else:
        return 0.0001 * (200-epoch) / 50

     #callbacks=[keras.callbacks.LearningRateScheduler(scheduler)]

