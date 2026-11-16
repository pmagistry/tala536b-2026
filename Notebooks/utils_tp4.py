from pathlib import Path
import json
import re
import os
import random
from collections import Counter

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "torch"

def lower_and_filter_names(txt: str) -> str:
    """
    Fonction de prétraitement du texte
    """
    txt = re.sub("\\p{P}", " ", txt) # enlever la ponctuation
    txt = " ".join(txt.lower().split())
    for name in ["figaro", "libération", "france-info", "bfmtv", "bfm"]:
        txt = txt.replace(name, "") # supprime quelques indices trop forts
    return " ".join(txt.split())

def load_data(corpus_file="./corpus/corpus-classif.json", target="catégorie") -> dict:
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

class PyTextVectorization:
    def __init__(self, vocab_size=1000, sequence_length=30):
        self.max_tokens = vocab_size
        self.sequence_length = sequence_length

        self.word_to_idx = {"[PAD]" : 0, "[UNK]" : 1} # 0 : padding (séquence fixe), 1 : mot inconnu, 2 : vocabulaire
        self.idx_to_word = ["[PAD]", "[UNK]"]

    def adapt(self, texts):
        counter = Counter()
        for text in texts : 
            counter.update(text.split())
        most_common = counter.most_common(self.vocab_size -2) # On enlève les PAD et UNK de la liste
        for word, _ in most_common :
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word.append(word)
            
    def __call__(self, texts):
        batch_size = []
        for text in texts:
            tokens = text.split()
            indices = [self.max_tokens(token, 1) for token in tokens]
            indices = indices[:self.sequence_length]

            sequence += [0] * (self.sequence_length - len(sequence)) # Padding vers la droite pour avoir des séquences de textes de même longueure
        return torch.tensor(sequences, dtype=torch.long)

    def get_vocabulary(self):
        return self.idx_to_word
    
    def vocabulary_size(self):
        return len(self.idx_to_word)


def create_textvectorizer(corpus, vocab_size=1000):
    tv = TextVectorization(vocab_size=vocab_size, sequence_length=30)
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
        texts, 
        y, 
        test_size=0.30, 
        stratify=y, 
        random_state=seed
    )
    x_dev, x_te, y_dev, y_te = train_test_split(
        x_tmp, 
        y_tmp, 
        test_size=0.50, 
        stratify=y_tmp, 
        random_state=seed
    )

    return (
        (x_tr, y_tr), 
        (x_dev, y_dev), 
        (x_te, y_te)
    )


def encode(subcorpus, tv, encodeur):
    X = tv(subcorpus[0])
    Y = torch.tensor([encodeur[y] for y in subcorpus[1]], dtype=torch.long)
    return X, Y


class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = y

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def create_dataloader(X, Y, batch_size=32, shuffle=False):
    dataset = TextDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class SequentialTextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=200, nb_classes=6, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(emb_size, nb_classes, bias=False)
    
    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1) # ignorer le padding
        emb = emb.marked_fill(~mark, float("-inf") )
        pooled = torch.max(emb, dim=1).values # GlobalMaxPooling1D
        pooled = torch.nan_to_num(pooled, neginf=0.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

class ResidualBlock(nn.Module):
    def __init__(self, emb_size, dropout=0.5):
        super().__init__()
        self.linear = nn.Linear(emb_size, emb_size, bias=False)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        hidden = self.linear(x)
        hidden = self.activation(hidden)
        hidden = hidden + x # connexion résiduelle
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        return hidden

class DeepTextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=200, nb_classes=6, dropout=0.5, nb_blocks=5):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=0)
        self.initial_norm = nn.LayerNorm(emb_size)
        self.blocls = nn.ModuleList([ResidualBlock(emb_size, dropout) for _ in rang(nb_blocks)])
        self.classifier = nn.Linear(emb_size, nb_classes, bias=False)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = emb.sum(dim=1) / lengths
        hidden = self.initial_norm(pooled) # première normalisation
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(hidden)

def create_model_seq(nb_input, vocab: list[str], emb_size=200, nb_classes=6, dropout=0.5, optim="sgd"):
    """
        Modèle simple, "séquentiel"
    """
    model = SequentialTextClassifier(
        vocab_size=len(vocab), 
        emb_size=emb_size, 
        nb_classes=nb_classes, 
        dropout=dropout)
    
    return model.to(DEVICE)

def create_model_func(nb_input, vocab: list[str], emb_size=200, nb_classes=6, dropout=0.5, optim="sgd"):
    """
        Example de modèle plus profond avec connexions résiduelles
        (à visualiser avec keras.utils.plot_model)
        Codé dans le style "functionnel"
    """
    model = DeepTextClassifier(
        vocab_size=len(vocab), 
        emb_size=emb_size, 
        nb_classes=nb_classes, 
        dropout=dropout
        nb_blocks=5
        )
    print(model)
    return model.to(DEVICE)

class PretrainedInitializer:
    def __init__(self, data_file:str, vocab:list[str]):
        self.data_file = data_file
        self.vocab = vocab
        self.w2v = KeyedVectors.load_word2vec_format(data_file, binary=true, unicode_errors="ignore")
    
    def get_weights(self, emb_size=200):
        # ignitialisation aléatoire
        weights = torch.empty(
            len(self.vocab),
            emb_size)
        nn.init.xavier_uniform_(weights)
        weights[0]=0 # padding = 0
        nb_found = 0
        for i, word, in enumerate(self.vocab):
            if word in self.w2v:
                vector = self.w2v[word]
                if len(vector) == emb_size:
                    weights[i]=torch.tensor(vector, dtype=torch.float32)
                    nb_found += 1
        print(f"{nb_found}/{len(self.vocab)} "
            "mots initialisés avec Word2Vec")
        return weights

def initialize_embeddings(model, data_file, vocab):
    initializer = PretrainedInitializer(data_file, vocab)
    weights = initializer.get_weights(emb_size=model.embedding.embedding_dim)
    model.embedding.weights.data.copy(weights.to(DEVICE))
    return model

## Creation des optimiseurs : SGD, Adam, RMSprop
def create_optimizer(model, optim="sgd", lr=0.001):
    optim = optim.lower()
    if optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else :
        raise ValueError(f"Optimiseur inconnu : {optim}")
    return optimizer

def scheduler(epoch, lr=None):
    if epoch < 50:
        return (0.0001 * (epoch+1)/ 50)
    elif epoch < 150:
        return 0.0001
    else:
        return 0.0001 * (200-epoch) / 50

def train_model(model, train_loader, valid_loader, epochs=200, optim="sgd", lr=0.001, use_scheduler=True):
    criterion = create_optimizer(model, optim=optim, lr=lr)
    if use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch=scheduler(epoch) / 0.0001)
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
    }
    for epoch in range(epochs):
        # Entrainement
        model.train()
        running_loss=0.0
        correct = 0
        total = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == Y_batch).sum().item()
            total += Y_batch.size(0)
        train_loss = running_loss / total
        train_accuracy = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss = loss.item() * X_batch.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == Y_batch).sum().item()
                val_total += Y_batch.size(0)
        val_loss /= val_total
        val_accuracy = val_correct / val_total

        # Historique 
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if use_scheduler:
            lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Accuracy: {train_accuracy:.4f} "
            f"Val Accuracy: {val_accuracy:.4f} "
            f"LR: {current_lr:.7f}"
        )

    return history

def predict(model, X, batch_size=32):
    model.eval()
    predictions = []
    loader = DataLoader(X, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X_batch in loader :
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            predictions.append(outputs.cpu())
    return torch.cat(predictions, dim=0)

def evaluate_mode(model, X_test, Y_test):
    outputs = predict(model, X_test, Y_test)
    Y_pred = outputs.argmax(dim=1)
    accuracy = accuracy_score(
        Y_test.np(),
        Y_pred.np()
    )
    print(f"Test accuracy : {accuracy:.4f}")
    return Y_pred

def plot_history(history):
    """
        fonction d'affichage de l'évolution de l'exactitude
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        history['accuracy'], 
        label='Training Accuracy')
    axes[0].plot(
        history['val_accuracy'], 
        label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history["loss"], 
        label="Training Loss"
    )
    axes[1].plot(
        history["val_loss"],
        label = "Validation Loss"
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_confusion(model, X_test, Y_test, labels):
    outputs = predict(model, X_test)
    Y_pred = outputs.argmax(dim=1).np()

    m = ConfusionMatrixDisplay.from_predictions(
        Y_test.np(), # en "sparse", Y est déjà bien encodé
        Y_pred 
        labels=range(len(labels)), 
        display_labels=labels,
        xticksrotation=45
    )
    plt.tight_layout()
    plt.show()
    return m


def plot_embeddings(model, tv, layer_idx=1):
    """
        affichage interactif de plongements lexicaux
        va chercher les plongements à la `layer_idx`-eme couche du modèle
        (on pourrait utiliser les argument name= pour la retrouver plus facilement)
    """
    import plotly.express as px
    from umap import UMAP
    _N_min=100
    _N_max=1000

    embeddings = (model.embedding.weights.detach().cpu().np())

    vocabulary = tv.get_vocabulary()

    umap = UMAP(
        n_components=2, 
        n_neighbors=10, 
        min_dist=0.2, 
        metric='cosine', 
        low_memory=False)

    embeddings_2d = umap.fit_transform(embeddings)

    df_embeddings = pl.DataFrame({
        'x': embeddings_2d[_N_min:_N_max, 0],
        'y': embeddings_2d[_N_min:_N_max, 1],
        'word': vocabulary[_N_min:_N_max]
    })

    fig = px.scatter(
        df_embeddings,
        x='x',
        y='y',
        text='word',
        title='UMAP Visualization of Embeddings from the First Layer (Interactive)',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        hover_name='word'
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.5),
        textposition='top center'
    )

    fig.update_layout(
        height=800,
        width=800,
        showlegend=False,
        hovermode='closest'
    )

    return fig

