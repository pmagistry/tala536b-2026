from collections import Counter
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

UPOS = 3
XPOS = 4

def read_conllu(path, column=UPOS):
    sentences, labels = [], []
    words, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            words.append(cols[1])
            tags.append(cols[column])
    if words:
        sentences.append(words)
        labels.append(tags)
    return sentences, labels

def count_tags(labels):
    return Counter(tag for tags in labels for tag in tags)

def build_vocabs(sentences, labels):
    word_to_num = {"<PAD>": 0, "<UNK>": 1}
    tag_to_num = {"<PAD>": 0, "<UNK>": 1}
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_num:
                word_to_num[word] = len(word_to_num)
    for label in labels:
        for tag in label:
            if tag not in tag_to_num:
                tag_to_num[tag] = len(tag_to_num)
    return word_to_num, tag_to_num

def encode(sentences, encoding):
    sentences_encoded = []
    for sentence in sentences:
        sentence_encoded = []
        for word in sentence:
            code = encoding[word] if word in encoding else encoding["<UNK>"]
            sentence_encoded.append(code)
        sentences_encoded.append(sentence_encoded)
    return sentences_encoded

def padding(sentences_enc, max_len):
    sentences_padded = np.zeros((len(sentences_enc), max_len), dtype=np.int32)
    for i, sentence_enc in enumerate(sentences_enc):
        for j, word_enc in enumerate(sentence_enc[:max_len]):
            sentences_padded[i, j] = word_enc
    return sentences_padded

def build_model(vocab_size, n_tags, max_len, emb_dim=100, lstm_units=128):
    model = keras.Sequential([
        keras.layers.Input(shape=(max_len,), dtype="int32"),
        keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True),
        keras.layers.LSTM(lstm_units, return_sequences=True),
        keras.layers.Dense(n_tags, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

