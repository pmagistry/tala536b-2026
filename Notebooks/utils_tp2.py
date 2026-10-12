from collections import Counter

import numpy as np


def load_data(path):
    texts = []
    labels = []
    with open(path) as file:
        for line in file:
            label, text = line.strip().split(" ", 1)
            labels.append(label)
            texts.append(text)
    return texts, labels


def bigrams(text):
    text = text.lower()
    return [text[i:i+2] for i in range(len(text) - 1)]


def get_vocab(texts, k=100):
    c = Counter()
    for text in texts:
        c.update(bigrams(text))
    return [bg for bg, n in c.most_common(k)]


def vectorise(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        bgs = bigrams(text)
        c = Counter(bgs)
        for j, bg in enumerate(vocab):
            X[i, j] = c[bg] / len(bgs)
    return X


def normalise_data(X):
    for k in range(X.shape[1]):
        mean = np.mean(X[:, k])
        std = np.std(X[:, k])
        X[:, k] = (X[:, k] - mean) / std
