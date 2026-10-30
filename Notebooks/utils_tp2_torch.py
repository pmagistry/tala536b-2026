from pathlib import Path
import numpy as np
import torch
import json
import numpy as np
import torch.nn as nn
from collections import Counter
import re


def load_data(corpus_path: str, limit=50) -> list[dict]:
    """
    charge un corpus de classification anglais/allemand
    """
    data = []
    with open(corpus_path) as file:
        for line in file:
            label, text = line.strip().split(" ", 1)
            n_th = text.count("th")
            n_en = text.count("en")
            n_le = text.count("le")
            n_st = text.count("st")
            n_tr = text.count("tr")
            n_sc = text.count("sc")
            l = len(text)
            
            instance = {
                "label": label, 
                "th": n_th / l, 
                "le": n_le / l, 
                "st": n_st / l, 
                "tr": n_tr / l, 
                "sc": n_sc / l, 
                "en": n_en / l
            }
            
            
            data.append(instance)
            
            if len(data) >= limit:
                return data
                
    return data

def load_json(corpus_path: str) -> list[dict]:
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

def normalise_data(dataset):
    """Centre et réduit les caractéristiques numériques du dataset."""
    for k in dataset[0].keys():
        if k != 'label':
            values = [d[k] for d in dataset]
            mean, std = np.mean(values), np.std(values)
            for d in dataset:
                d[k] = (d[k] - mean) / std

def encode_corpus_one_hot(dataset: list[dict], langues: list[str]) -> tuple[np.array, np.array]:
    bigrams_keys = list(sorted(k for k in dataset[0].keys() if k != 'label'))
    X_lang = np.array([[d[k] for k in bigrams_keys] for d in dataset])
    y_lang = np.array([[1 if d['label'] == l else 0 for l in langues] for d in dataset])
    return X_lang, y_lang


def encode_corpus_sparse(dataset: list[dict], langues: list[str]) -> tuple[np.array, np.array]:
    bigrams_keys = list(sorted(k for k in dataset[0].keys() if k != 'label'))
    X_lang = np.array([[d[k] for k in bigrams_keys] for d in dataset])
    code_langues = {l: i for i, l in enumerate(langues)}
    y_lang = np.array([code_langues[d['label']] for d in dataset])
    return X_lang, y_lang


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


class MLPSequential(nn.Module):
    """Petit MLP binary (équivalent de keras_mlp_sequential)."""

    def __init__(self, nb_input=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nb_input, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def torch_mlp_sequential(nb_input=2):
    return MLPSequential(nb_input)
  
#modification de l'architechture du script afin que le model puisse apprendre a categorisé les articles.

def construire_vocab(textes, max_tokens=5000):
    """Équivalent de TextVectorization.adapt : 0 = padding, 1 = inconnu."""
    compteur = Counter(mot for t in textes for mot in re.findall(r"\w+", t.lower()))
    mots = [m for m, _ in compteur.most_common(max_tokens - 2)]
    return {m: i + 2 for i, m in enumerate(mots)}
 
 
def vectoriser(textes, vocab, longueur=500):
    """Équivalent de TextVectorization(output_mode="int").
    Renvoie un tableau (N, longueur) d'indices entiers."""
    X = np.zeros((len(textes), longueur), dtype=np.int64)
    for i, t in enumerate(textes):
        ids = [vocab.get(m, 1) for m in re.findall(r"\w+", t.lower())[:longueur]]
        X[i, :len(ids)] = ids
    return X
 
 
def creer_modele_(vocab, nb_classes=7, dim=64, dropout=0.3):
    """Renvoie (embedding, tete): les deux blocs qui contiennent des poids appris."""
    embedding = nn.Embedding(len(vocab) + 2, dim, padding_idx=0)
    tete = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(dim, 64), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, nb_classes),
      #produit 7 scores, un score par catégorie.
    )
    return embedding, tete
 
 
def predire_(embedding, tete, X, pooling="avg"):
    """(N, 500) indices -> (N, nb_classes) logits.
    pooling="avg" : GlobalAveragePooling1D ; pooling="max" : GlobalMaxPooling1D."""
  
    X = torch.as_tensor(np.asarray(X), dtype=torch.long)
    emb = embedding(X)                        
  # (N, 500, dim)
    masque = (X != 0).unsqueeze(-1)           
  # ignore le padding (mask_zero=True)
    if pooling == "avg":
        vecteur = (emb * masque).sum(1) / masque.sum(1).clamp(min=1)
    else:
        vecteur = emb.masked_fill(~masque, -1e9).max(dim=1).values
      
    return tete(vecteur)  # (N, nb_classes)
 
 
def accuracy_(embedding, tete, X, Y, pooling="avg"):

    embedding.eval()
    tete.eval()
    Y = torch.as_tensor(np.asarray(Y), dtype=torch.long)
  
    with torch.no_grad():
        logits = predire_(embedding, tete, X, pooling)
      
    return (logits.argmax(dim=1) == Y).float().mean().item()
 


def entrainer_(embedding, tete, X_train, Y_train, X_test=None, Y_test=None,
                        pooling="avg", epochs=30, batch_size=32, lr=1e-3):
    """Boucle d'entraînement (Adam + CrossEntropyLoss).
    Renvoie un historique {"loss": [...], "train": [...], "test": [...]}."""
    X_train = torch.as_tensor(np.asarray(X_train), dtype=torch.long)
    Y_train = torch.as_tensor(np.asarray(Y_train), dtype=torch.long)
    params = list(embedding.parameters()) + list(tete.parameters())
    optimiseur = torch.optim.Adam(params, lr=lr)
    perte = nn.CrossEntropyLoss()
    n = len(X_train)
    historique = {"loss": [], "train": [], "test": []}
  #Après chaque epoch, on mesure l’accuracy sur le train et sur le test, et on la range dans historique pour tracer les courbes.
    for epoch in range(epochs):
        embedding.train()
        tete.train()
        ordre = torch.randperm(n) 
      #mélange l’ordre des textes, 
      #différemment à chaque epoch.
        for i in range(0, n, batch_size):
            idx = ordre[i:i + batch_size]
            optimiseur.zero_grad()
            loss = perte(predire_(embedding, tete, X_train[idx], pooling), Y_train[idx])
            loss.backward() #calcule, pour chaque poids, dans quel sens il faut le modifier pour réduire l’erreur.
            optimiseur.step()#applique la modification.
        acc_train = accuracy_(embedding, tete, X_train, Y_train, pooling)
        historique["loss"].append(loss.item())
        historique["train"].append(acc_train)
        msg = f"epoch {epoch + 1}/{epochs} - loss: {loss.item():.4f} - train: {acc_train:.3f}"
        if X_test is not None:
            acc_test = accuracy_(embedding, tete, X_test, Y_test, pooling)
            historique["test"].append(acc_test)
            msg += f" - test: {acc_test:.3f}"
        print(msg)
    return historique