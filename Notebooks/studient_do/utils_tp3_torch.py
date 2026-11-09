import json
import regex
import os
import numpy as np
import torch
import torch.nn as nn
from keras import layers
import matplotlib.pyplot as plt
from typing import Literal
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Prétraitement du corpus gardé
# ------------------------------

def lower_and_filter_names(txt: str) -> str:
    """
    Fonction de prétraitement du texte
    """
    txt = regex.sub("\\p{P}", " ", txt) # enlever la ponctuation
    txt = " ".join(txt.lower().split())
    for name in ["figaro", "libération", "france-info", "bfmtv", "bfm"]:
        txt = txt.replace(name, "") # supprime quelques indices trop forts
    return txt


def load_data(corpus_file="../../corpus/corpus-classif.json", target="catégorie") -> dict:
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


def create_textvectorizer(corpus, vocab_size=1000):
    tv = layers.TextVectorization(
        output_sequence_length=30,
        max_tokens=vocab_size,
        standardize=None,  # si c'est fait en amont
        split="whitespace",
        ngrams=None,
        output_mode="int",
    )
    tv.adapt([line['text'] for line in corpus])
    return tv

def stratified_split(corpus, seed=42):
    """
        retourne trois tuples: train, valid, test
        chaque sous-corpus est un tuple (X,Y)
    """
    texts = [r["text"] for r in corpus]
    y = [r["label"] for r in corpus]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        texts, y, test_size=0.30, stratify=y, random_state=seed
    )
    x_dev, x_te, y_dev, y_te = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return (x_tr, y_tr), (x_dev, y_dev), (x_te, y_te)


def encode(subcorpus, tv, encodeur):
    X = tv(subcorpus[0]).cpu().numpy()
    Y = np.array([encodeur[y] for y in subcorpus[1]])

    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# Modèle 1
# --------

class ModelSeq(nn.Module) : 

    def __init__(self, vocab_size, emb_size=200, nb_classes=6, dropout=0.5):
        super().__init__() 
        self.embbeding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(emb_size, nb_classes, bias=False)

    def forward(self, x) :
        x = self.embbeding(x)
        x = torch.max(x, dim=1).values() # GlobalMaxPooling1D
        x = self.dropout(x)
        x = self.output(x)
        return x

# Méthode 2
# ---------

class ResidualBlock(nn.Module) :

    def __init__(self, emb_size, dropout):
        super().__init__()
        self.dense = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) :
        residual = x
        x = nn.functional.relu(self.dense(x))
        x = residual + x
        x = self.dropout(x)
        return x

class ModelFunc(nn.Module) :
    def __init__(self, vocab_size, emb_size=200, nb_classes=6, dropout=0.5, nb_block=5):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        # GlobalAveragePooling1D
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Blocs résiduels
        self.blocks = nn.ModuleList([ResidualBlock(emb_size, dropout) for _ in range(nb_block)])
        # Classification
        self.output = nn.Linear(emb_size, nb_classes, bias=False) 

    def forward(self, x) :
        x = self.embedding(x)
        # GlobalAveragePooling1D
        x = x.mean(dim=1)

        for block in self.blocks :
            x = block(x)

        x = self.output(x)
        return x

from gensim.models.keyedvectors import KeyedVectors

def initialize_embeddings(embedding_layer, vocab, data_file) :
    w2v = KeyedVectors.load_word2vec_format(data_file, binary=True, unicode_errors="ignore")
    emb_size = embedding_layer.embedding_dim

    weights = np.random.uniform(
        low=np.sqrt(6/(len(vocab) + emb_size)), 
        high=np.sqrt(6/(len(vocab) + emb_size)), 
        size=(len(vocab), emb_size)).astype(np.float32)
    found = 0

    for i, word in enumerate(vocab) :
        if word in w2v :
            weights[i] = w2v[word]
            found += 1
    with torch.no_grad():
        embedding_layer.weight.copy_(torch.tensor(weights))
    print(f"{found}/{len(vocab)} mots initialisés avec Word2Vec")
    return embedding_layer

def train_model(model, train_loader, dev_loader, epochs=200, lr=0.0001, optimizer_name:Literal["sdg", "adam", "rmsprop"]="sdg") :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        "sdg" : torch.optim.SGD,
        "adam" : torch.optim.Adam,
        "rmsprop" : torch.optim.RMSprop
    }
    optimizer = optimizers[optimizer_name.lower()](model.parameters(), lr=lr)
    history = {
        "loss" : [],
        "accuracy" : [],
        "val_loss" : [],
        "val_accuracy" : []
    }

    for epoch in range(epochs) :
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X, Y in train_loader :
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*X.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == Y).sum().item()
            total += Y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad() :
            for X, Y in dev_loader :
                X = X.to(device)
                Y = Y.to(device)
                outputs = model(X)
                loss = criterion(outputs, Y)
                val_loss += loss.item()*X.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == Y).sum().item()
                val_total += Y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs}"
            f"\tloss : {train_loss:.4f}"
            f"\taccuracy : {train_acc:.4f}"
            f"\tval_loss : {val_loss:.4f}"
            f"\tval_accuracy : {val_acc:.4f}"
              )

    return model, history

def scheduler(epoch, lr) :
    if epoch < 150 :
        return (0.0001 * (epoch+1)/50)
    elif epoch < 150 :
        return 0.0001
    else :
        return 0.0001 * (200-epoch) / 50

def plot_history(history) :
    fig = plt.figure(figsize=(10,6))

    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)
    return fig

def plot_confusion(model, X_test, Y_test, labels) :
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        outputs = model(X_test.to(device))
        Y_pred = outputs.argmax(dim=1).cpu().numpy()

    Y_true = Y_test.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay.from_predictions(
        Y_true,
        Y_pred, 
        labels=range(len(labels)),
        display_labels=labels,
        ax=ax
    )
    plt.tight_layout()
    return fig

def plot_embeddings(model, vocab) :
    import plotly.express as px
    import polars as pl
    from umap import UMAP
    N_min = 100
    N_max = 1000

    embeddings = (model.embedding.weight.detach().cpu().numpy())
    umap = UMAP(n_components=2, n_neighbors=10, min_dist=0.2, metric="cosine", low_memory=False)
    embeddings_2d = umap.fit_transform(embeddings)
    df_embeddings = pl.DataFrame({
        "x" : embeddings_2d[N_min:N_max, 0],
        "y" : embeddings_2d[N_min:N_max, 1],
        "word" : vocab[N_min:N_max]
    })

    fig = px.scatter(df_embeddings, x="x", y="y", text="word", title="UMAP Visualization of Embeddings", hover_name="word")
    fig.update_layout(height=800, width=800, showlegend=False)
    return fig