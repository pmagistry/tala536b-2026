from pathlib import Path
import numpy as np
import polars as pl
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import json
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

def load_json(corpus_path: str) -> list[dict]:
    """
        charge un corpus de classification au format JSON
        (liste d'objets avec les clefs "texte" et "catégorie")
        produit le même format que load_text : clefs label et text
    """
    with open(corpus_path, encoding="utf-8") as file:
        brut = json.load(file)

    data = []
    for d in brut: 
      if d["catégorie"] == '' or not d["texte"] or not d["texte"].strip():
        continue
      data.append({"label": d["catégorie"] , "text":d["texte"] })
      
        
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


def create_model(vocab_size, nb_classes, dim=64, optim="Adam", dropout=0.3):
    model = Sequential([
        keras.layers.Input(shape=(5000,)),
        keras.layers.Embedding(vocab_size, dim, mask_zero=True),
        keras.layers.GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        keras.layers.Dropout(dropout),
        Dense(nb_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=optim,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
