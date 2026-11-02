import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# torch.manual_seed(0)
from pathlib import Path
import polars as pl
import os
import json

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.models import Sequential
from keras.layers import Dense, Normalization, TextVectorization
from keras.optimizers import SGD
import marimo as mo

# chargement des données

def load_data(path, bigramms, mapping):

    data = []

    with open(path, 'r') as f:
        for line in f :
            label, text = line.strip().split(' ', maxsplit=1)
            instance = {'label' : mapping[label]}
            counts = {e : text.count(e) for e in bigramms}
            instance.update(counts)
            data.append(instance)

    return data

def load_text(path):
    data = []
    with open(path, 'r') as f:
        for line in f :
            label, text = line.strip().split(' ', maxsplit=1)
            instance = {'label' : label, 'text' : text}
            data.append(instance)
    return data

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return [{'label' : d['catégorie'], 'text' : d['texte']} for d in data if d['texte'] !='\n' and d['catégorie'] != '']


# modèles

def model_binary_bigrams(hidden_neurons, n_arguments, norm=True):
    model = Sequential([
        keras.layers.Input(shape=(n_arguments,)),
        Normalization() if not norm else Dense(hidden_neurons, activation='relu'),
        Dense(hidden_neurons, activation="relu"),
        Dense(1, activation ='sigmoid')
    ])

    model.compile(
        optimizer=SGD(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def model_multiclass_categorical_bigramms(num_cats, hidden_neurons, n_arguments, norm=True) :
    model = Sequential([
        keras.layers.Input(shape=(n_arguments,)),
        Normalization() if not norm else Dense(hidden_neurons, activation='relu'),
        Dense(hidden_neurons, activation="relu"),
        Dense(num_cats, activation='softmax')
    ])

    model.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def model_multiclass_sparse_bigramms(num_cats, hidden_neurons, n_arguments, norm=True):
    model = Sequential([
        keras.layers.Input(shape=(n_arguments,)),
        Normalization() if not norm else Dense(hidden_neurons, activation='relu'),
        Dense(hidden_neurons, activation="relu"),
        Dense(num_cats, use_bias=False,activation='softmax')
    ])

    model.compile(
        optimizer=SGD(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def vectorizer(corpus):
    train, test = train_test_split(corpus, train_size=0.8, random_state=42)
    vectorizer_layer = keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        output_mode='count',
        encoding='utf8')

    vectorizer_layer.adapt(c['text'] for c in train)
    return vectorizer_layer, test, train

def model_vectorizer(num_cats, hidden_neurons, n_arguments) :

    model = Sequential([keras.layers.Input(shape=(n_arguments,)),
                        Dense(hidden_neurons, activation='relu'),
                        Dense(num_cats, activation='softmax')])

    model.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def model_embedding(vectorizer, num_cats, hidden_neurons, n_arguments) :

    vocab = vectorizer.get_vocabulary()

    pool = keras.layers.GlobalAveragePooling1D()

    model = Sequential([keras.layers.Input(shape=(n_arguments,)),
                        keras.layers.Embedding(input_dim=len(vocab), output_dim=hidden_neurons),
                        pool,
                        Dense(hidden_neurons, activation='relu'),
                        keras.layers.Dropout(0.5),
                        Dense(num_cats, activation='softmax')
    ])
    model.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# fonctions utilitaires

def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != 'label':
            mean = np.mean([d[k] for d in dataset])
            std = np.std([d[k] for d in dataset])
            for d in dataset:
                d[k] = (d[k] - mean) / std


def create_cat_onehot(instances, labels, n_cat,mapping=False):
    categories = []

    mapping_cats = { l : i for i, l in enumerate(labels)}

    if not mapping :
        for i in instances :
            c = np.zeros((n_cat,), dtype=float)
            c[[i['label']]] = 1.0
            categories.append(c)
    else :
        for i in instances :
            c = np.zeros((n_cat,), dtype=float)
            c[mapping_cats[i['label']]] = 1.0
            categories.append(c)

    return np.array(categories)

def create_cat_sparse(instances, labels):

    mapping = { l : i for i, l in enumerate(labels)}

    c = np.array([mapping[i['label']] for i in instances])

    return c

def format_prediction_onehot(prediction, labels):

    values = prediction.tolist()

    index_highest = values.index(max(values))

    l = labels[index_highest]

    return l


def format_data(dataset, labels, method):
    keys = list(sorted(k for k in dataset[0].keys() if k != 'label'))  # Tout ce qui n'est pas label doit être un code de langue

    if method == 'vectorized' :
        return create_cat_onehot(dataset, labels, len(labels), mapping=True)

    x = np.array([[d[k] for k in keys] for d in dataset])

    if method == 'onehot':
        y = create_cat_onehot(dataset, labels, len(keys))
    elif method == 'sparse':
        y = create_cat_sparse(dataset, labels)
    elif method == 'predict':
        return x
    else :
        print('Choose a valid method')
        return None

    return x, y
