import numpy as np
import json
import regex
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# j'ai adapté sans trop de problèmes les fonctions de prétraitement,
# j'en ai réjouté pour créer un vocabulaire et le word2Index,
# j'avais à peu près quelque chose de convaincant pour la vectorisation mais j'ai pas eu l'intuition
# de faire par batch pour résoudre le problème de padding des phrases que j'avais
# pour le reste, c'est essentiellement généré par GLM 5.3 Flash (via mistral)
# j'ai essayé de comprendre autant que possible mais ça reste cryptique
# je comprends les entrées et les sorties de chaque fonction mais tout ce qu'il y a entre me reste opaque
# mais voilà, disclaimer pour dire que je n'ai clairement pas compris tout ce qui est écrit ici

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

def create_vocab(corpus):
    vocab = ["[PAD]", "[OOV]"] + sorted(set(" ".join([sent["text"] for sent in corpus]).split()))
    return vocab

def create_word2Index(vocab):
    word2Index = {word:i for i, word in enumerate(vocab)}
    return word2Index

def encode_batch(sentences, word2Index):
    seqs = [
        [word2Index.get(w, word2Index["[OOV]"]) for w in s["text"].split()]
        for s in sentences
    ]
    lengths = torch.tensor([len(s) for s in seqs])
    max_len = lengths.max().item()

    padded = torch.full((len(seqs), max_len), word2Index["[PAD]"], dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)

    return padded, lengths


def _fit(model, train, Y_train, word2Index, epochs=100, batch_size=32, lr=0.01):
    Y = torch.as_tensor(np.asarray(Y_train), dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(train)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, lengths = encode_batch([train[j] for j in idx], word2Index)
            yb = Y[idx]
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == yb).sum().item()
        print(f"epoch {epoch + 1}/{epochs} - loss: {total_loss / n:.4f} - accuracy: {correct / n:.4f}")
    return model

def model_embedding(vocab_size, embed_dim=50, nb_classes=6, lr=1e-3):
    return NNembedding(vocab_size, embed_dim, nb_classes), lr

def train_model(model, loss, X_train, Y_train, epochs=100, batch_size=32):
    return _fit(model, X_train, Y_train, loss, epochs=epochs, batch_size=batch_size)


def evaluate_model(model, test, Y_test, word2Index, batch_size=64):
    Y = torch.as_tensor(np.asarray(Y_test), dtype=torch.long)
    model.eval()
    total_loss, correct, n = 0.0, 0, len(test)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb, _ = encode_batch(test[i:i + batch_size], word2Index)
            yb = Y[i:i + batch_size]
            logits = model(xb)
            total_loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
    print(f"loss: {total_loss / n:.4f} - accuracy: {correct / n:.4f}")
    return total_loss / n, correct / n


def predict_model(model, sentences, word2Index, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            xb, _ = encode_batch(sentences[i:i + batch_size], word2Index)
            logits = model(xb)
            preds.append(logits.argmax(dim=1))
    return torch.cat(preds).numpy()


class NNembedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, nb_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(embed_dim, nb_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = self.dropout(x)
        bs, _, e = x.shape
        x = F.adaptive_max_pool1d(x.transpose(1, 2), 1).reshape(bs, e)
        return self.linear(x)
