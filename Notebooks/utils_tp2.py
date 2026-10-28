from pathlib import Path
import numpy as np
import polars as pl
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.models import Sequential
from keras.layers import Dense, Normalization
from keras.optimizers import SGD


def load_data(corpus_path: str, limit=50) -> list[dict]:
    """
        charge un corpus de classification anglais/allemand
    """
    data = []
    with open(corpus_path) as file:
        for line in file:
            label, text = line.strip().split(" ",1)
            n_th = text.count("th")
            n_en = text.count("en")
            n_le = text.count("le")
            n_st = text.count("st")
            n_tr = text.count("tr")
            n_sc = text.count("sc")
            l = len(text)
            instance = {"label":label, "th": n_th / l, "en": n_en /l}
            data.append(instance)
            if len(data) >= limit:
                return data
    return data


def load_text(corpus_path: str, limit=50) -> list[dict]:
    """
        charge un corpus de classification
        produit un dictionnaire avec les clefs label et text
        à utiliser avec une TextVectorization
    """
    data = []
    with open(corpus_path) as file:
        for line in file:
            label, text = line.strip().split(" ",1)
            instance = {"label":label, "text": text}
            data.append(instance)
            if len(data) >= limit:
                return data
    return data


def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != 'label':
            mean = np.mean([d[k] for d in dataset])
            std = np.std([d[k] for d in dataset])
            for d in dataset:
                d[k] = (d[k]- mean) / std


def encode_corpus_one_hot(dataset: list[dict], langues: list[str]) -> tuple[np.array, np.array]:
    bigrams_keys = list(sorted( k for k in dataset[0].keys() if k != 'label'))  # Tout ce qui n'est pas label doit être un code de langue
    X_lang = np.array([[d[k] for k in bigrams_keys] for d in dataset])  #  l'encodage s'adaptera au nombre de bigrammes définis
    y_lang = np.array([[1 if d['label'] == l else 0 for l in langues] for d in dataset])
    return X_lang, y_lang


def encode_corpus_sparse(dataset: list[dict], langues: list[str]) -> tuple[np.array, np.array]:
    bigrams_keys = list(sorted( k for k in dataset[0].keys() if k != 'label'))  # Tout ce qui n'est pas label doit être un code de langue
    X_lang = np.array([[d[k] for k in bigrams_keys] for d in dataset])  #  l'encodage s'adaptera au nombre de bigrammes définis
    code_langues = {l:i for i, l in enumerate(langues)}
    y_lang = np.array([code_langues[d['label']] for d in dataset])
    return X_lang, y_lang


def model_sparse(nb_input=2, nb_classes=6, optim="sgd"):
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        Normalization(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(nb_classes, use_bias=False, activation='softmax')
    ])
    model.compile(
        optimizer=optim,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],

    )
    return model

def model_sparse_dropout(nb_input=2, nb_classes=6, rate=0.5, optim="sgd"):
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        Normalization(),
        Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        Dense(nb_classes, use_bias=False, activation='softmax')
    ])
    model.compile(
        optimizer=optim,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def model_embedding(vocab_size, nb_classes, seq_len=100, embedding_dim=32, pooling="average", optim="sgd"):
    if pooling == "average":
        couche_pooling = keras.layers.GlobalAveragePooling1D()
    else:
        couche_pooling = keras.layers.GlobalMaxPooling1D()

    model = Sequential([
        keras.layers.Input(shape=(seq_len,), dtype="int32"),
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        couche_pooling,
        Dense(nb_classes, activation='softmax')
    ])
    model.compile(
        optimizer=optim,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def model_one_hot(nb_input=2, nb_classes=6, optim="sgd"):
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        Normalization(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(nb_classes, use_bias=False, activation='softmax')
    ])
    model.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=['accuracy'],

    )
    return model
