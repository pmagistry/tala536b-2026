"""
Version PyTorch de utils_tp3 :
- les fonctions de données et d'affichage sont reprises de utils_tp3
- seuls l'encodage, le modèle, l'entraînement et l'évaluation sont réécrits
"""
from types import SimpleNamespace
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import utils_tp3
# fonctions reprises telles quelles
from utils_tp3 import load_data, create_textvectorizer, stratified_split, plot_history


def encode(subcorpus, tv, encodeur):
    """ comme utils_tp3.encode, mais renvoie des tenseurs PyTorch d'entiers """
    X, Y = utils_tp3.encode(subcorpus, tv, encodeur)
    X = torch.tensor(keras.ops.convert_to_numpy(X), dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


class ModelFunc(nn.Module):
    """
        Même architecture que utils_tp3.create_model_func :
        Embedding -> moyenne -> 5 blocs résiduels -> sortie
        (les Normalization() de Keras ne sont pas adaptées, elles ne font rien : on les omet)
    """
    def __init__(self, voc_size, emb_size, nb_classes, dropout, nb_blocks=5):
        super().__init__()
        self.emb = nn.Embedding(voc_size, emb_size)
        self.denses = nn.ModuleList(
            [nn.Linear(emb_size, emb_size, bias=False) for _ in range(nb_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_size, nb_classes, bias=False)

    def forward(self, inputs):
        hidden = self.emb(inputs).mean(dim=1)          # GlobalAveragePooling1D
        for dense in self.denses:
            hidden2 = torch.relu(dense(hidden))        # Dense(relu)
            hidden = self.dropout(hidden2 + hidden)    # Add + Dropout
        return self.out(hidden)                        # pas de softmax (cf. CrossEntropyLoss)


def create_model_func(nb_input, vocab: list[str], emb_size=200, nb_classes=6, dropout=0.5):
    """ renvoie le modèle et la fonction de coût (pas de compile en PyTorch) """
    model = ModelFunc(len(vocab), emb_size, nb_classes, dropout)
    loss = nn.CrossEntropyLoss()   # = softmax + sparse_categorical_crossentropy
    return model, loss


def evaluate_model(model, X, Y, verbose=True):
    """ équivalent de model.evaluate : renvoie [loss, accuracy] """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, Y).item()
        acc = (logits.argmax(dim=1) == Y).float().mean().item()
    if verbose:
        print(f"loss: {loss:.4f} - accuracy: {acc:.4f}")
    return [loss, acc]


def train_model(model, loss, X, Y, epochs=400, batch_size=32, validation_data=None,
                optim="adam", start_from_epoch=50, patience=30):
    """
        équivalent de model.fit(..., validation_data=...,
                                callbacks=[EarlyStopping("val_accuracy", start_from_epoch, patience)])
        renvoie un objet avec .history, comme Keras (pour réutiliser plot_history)
    """
    if optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    history = {"accuracy": [], "val_accuracy": []}
    best, wait = 0, 0
    for epoch in range(epochs):
        # --- entraînement ---
        model.train()
        perm = torch.randperm(len(X))                  # on mélange à chaque epoch
        correct = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X[idx])
            l = loss(logits, Y[idx])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == Y[idx]).sum().item()
        acc = correct / len(X)
        history["accuracy"].append(acc)

        if validation_data is None:
            print(f"Epoch {epoch+1}/{epochs} - accuracy: {acc:.4f}")
            continue

        # --- validation ---
        _, val_acc = evaluate_model(model, *validation_data, verbose=False)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - accuracy: {acc:.4f} - val_accuracy: {val_acc:.4f}")

        # --- early stopping ---
        if epoch >= start_from_epoch:
            if val_acc > best:
                best, wait = val_acc, 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping à l'epoch {epoch+1}")
                    break

    return SimpleNamespace(history=history)


def predict_model(model, X):
    """ équivalent de model.predict : probabilités en numpy """
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).numpy()


def plot_confusion(model, X_test, Y_test, labels):
    """ comme utils_tp3.plot_confusion, avec predict_model """
    Y_pred = predict_model(model, X_test)
    m = ConfusionMatrixDisplay.from_predictions(
        Y_test.numpy(),
        np.argmax(Y_pred, axis=1),
        labels=range(len(labels)),
        display_labels=labels
    )
    plt.show()
    return m


def plot_embeddings(model, tv):
    """ réutilise utils_tp3.plot_embeddings en lui présentant les poids comme Keras """
    weights = model.emb.weight.detach().numpy()
    fake_keras_model = SimpleNamespace(layers=[SimpleNamespace(get_weights=lambda: [weights])])
    return utils_tp3.plot_embeddings(fake_keras_model, tv, 0)
