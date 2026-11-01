import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.models import Sequential
from keras.layers import Dense, Normalization, Embedding, Dropout
import json

def load_text(corpus_path: str, limit=50) -> list[dict]:
    """
        charge un corpus JSON
    """
    data = []
    with open(corpus_path) as file:
        articles = json.load(file)

        for article in articles:
            text = article["texte"].strip()
            label = article["catégorie"]

            instance = {"label":label, "text": text}
            data.append(instance)

            if len(data) >= limit:
                return data
    return data

def model_embedding(nb_input=1000, nb_classes=6, vocab_size=1000, optim="sgd"):
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        #Normalization(), ==> on a des identifiants en entrée cette fois
        keras.layers.Embedding(vocab_size, 64),
        keras.layers.GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3), # désactive 30% du neurone
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(nb_classes, use_bias=False, activation='softmax')
    ])
    model.compile(
        optimizer=optim,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],

    )
    return model