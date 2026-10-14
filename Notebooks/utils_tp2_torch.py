from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


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


def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != 'label':
            mean = np.mean([d[k] for d in dataset])
            std = np.std([d[k] for d in dataset])
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
