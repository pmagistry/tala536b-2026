from pathlib import Path
import numpy as np
import json
import regex
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader


def lower_and_filter_names(txt: str) -> str:
    """
    Fonction de prétraitement du texte
    """
    txt = regex.sub("\\p{P}", " ", txt) # enlever la ponctuation
    txt = " ".join(txt.lower().split())
    for name in ["figaro", "libération", "france-info", "bfmtv", "bfm"]:
        txt = txt.replace(name, "") # supprime quelques indices trop forts
    return txt


def load_data_json(corpus_file="./corpus-classif.json", target="catégorie") -> dict:
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

def get_vocab(corpus):
    vocab = ["[PAD]"] + "[OOV]" + sorted(set(" ".join(corpus).split()))
    return vocab

def get_word2Index(vocab):
    word2Index = {word:i for i, word in enumerate(vocab)}
    return word2Index

def encode_corpus_one_hot(corpus: list[str], word2Index: dict):
    pass

def encode_corpus_sparse(corpus: list[str], encodeur):
    arr = []
    for sent in corpus:
        vect = []
        for i, word in enumerate(corpus):
            if word not in word2Index:
                vect.append(word2Index["[OOV]"])
            else:
                vect.apend(word2Index[word])
        arr.append(vect)
    return torch.tensor(arr, dtype=torch.int32), np.array(encodeur[sent[label]] for sent in corpus)

class Normalization(nn.Module):
    """Couche de normalisation (équivalent de keras.layers.Normalization)."""

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def adapt(self, data):
        data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        self.mean = data.mean(dim=0)
        std = data.std(dim=0)
        self.std = torch.where(std == 0, torch.ones_like(std), std)

    def forward(self, x):
        if self.mean is None:
            raise RuntimeError("Appelez adapt(data) avant l'entraînement.")
        return (x - self.mean) / self.std


class MLP6Langues(nn.Module):
    """Réseau dense : normalisation + 6 couches cachées relu + softmax."""

    def __init__(self, nb_input=2, nb_classes=6):
        super().__init__()
        self.normalization = Normalization()
        self.dense = nn.Sequential(
            nn.Linear(nb_input, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, nb_classes, bias=False),
        )

    def forward(self, x):
        x = self.normalization(x)
        return self.dense(x)


def _fit(model, X_train, Y_train, loss_fn, epochs=100, batch_size=32, lr=0.01):
    X = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(Y_train))
    if Y.ndim > 1:
        Y = Y.float()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]
            optimizer.zero_grad()
            logits = model(xb)
            if loss_fn.target_is_sparse:
                loss = loss_fn.fn(logits, yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
            else:
                loss = loss_fn.fn(logits, yb)
                correct += (logits.argmax(dim=1) == yb.argmax(dim=1)).sum().item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{epochs} - loss: {total_loss / n:.4f} - accuracy: {correct / n:.4f}")
    return model


class _SparseLoss:
    target_is_sparse = True
    fn = staticmethod(nn.CrossEntropyLoss())


class _OneHotLoss:
    target_is_sparse = False
    fn = staticmethod(nn.CrossEntropyLoss())


def model_sparse(nb_input=2, nb_classes=6, optim="sgd"):
    return MLP6Langues(nb_input, nb_classes), _SparseLoss()


def model_one_hot(nb_input=2, nb_classes=6, optim="sgd"):
    return MLP6Langues(nb_input, nb_classes), _OneHotLoss()


def train_model(model, loss, X_train, Y_train, epochs=100, batch_size=32):
    return _fit(model, X_train, Y_train, loss, epochs=epochs, batch_size=batch_size)


def evaluate_model(model, X_test, Y_test, loss=None):
    X = torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(Y_test))
    if Y.ndim > 1:
        Y = Y.float()
    sparse = Y.ndim == 1 or (loss is not None and loss.target_is_sparse)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss_value = nn.functional.cross_entropy(logits, Y).item()
        if sparse:
            acc = (logits.argmax(dim=1) == Y).float().mean().item()
        else:
            acc = (logits.argmax(dim=1) == Y.argmax(dim=1)).float().mean().item()
    print(f"loss: {loss_value:.4f} - accuracy: {acc:.4f}")
    return loss_value, acc


def predict_model(model, X):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).numpy()


class NNembedding(nn.Module):
    """Petit MLP binary (équivalent de keras_mlp_sequential)."""

    def __init__(self, vocab_size, max_len, embed_dim):
        super(NNembedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.linear = nn.Linear(max_len, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        bs, _, _ = x.shape
        x = F.adaptive_max_pool1d(x, 1).reshape(bs, -1)
        return self.linear(x)


def torch_mlp_sequential(nb_input=2):
    return MLPSequential(nb_input)
