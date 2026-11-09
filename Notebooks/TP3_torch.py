import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path
    from collections import Counter
    import regex
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report

    torch.manual_seed(42) 
    np.random.seed(42)
    return (
        ConfusionMatrixDisplay,
        Counter,
        Path,
        classification_report,
        json,
        nn,
        plt,
        regex,
        torch,
        train_test_split,
    )


@app.cell
def _(Path, json):
    corpus_path = Path.home() / "Downloads" / "corpus-classif.json"
    with corpus_path.open(encoding="utf-8") as f:
        articles = json.load(f)
    return (articles,)


@app.cell
def _(articles, regex):
    def clean_and_filter(raw_articles):
        processed = []
        for item in raw_articles:
            text = item["texte"]
            label = item["catégorie"]

            if len(text.strip().split()) <= 20 or not label:
                continue

            text = regex.sub(r"\p{P}", " ", text)  
            text = " ".join(text.lower().split())  
            for source_name in ["figaro", "libération", "france-info", "bfmtv", "bfm"]:
                text = text.replace(source_name, "")
            processed.append({"label": label, "text": text})
        return processed

    corpus = clean_and_filter(articles)
    labels = sorted({r["label"] for r in corpus})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return corpus, label_to_id, labels


@app.cell
def _(corpus, train_test_split):
    train, temporary = train_test_split(
        corpus,
        test_size=0.30,
        random_state=42,
        stratify=[r["label"] for r in corpus],
    )
    dev, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=42,
        stratify=[r["label"] for r in temporary],
    )
    print(f"train={len(train)}, validation={len(dev)}, test={len(test)}")
    return dev, test, train


@app.cell
def _(Counter, train):
    def build_vocab(training_articles, max_tokens=5000):
        counts = Counter()
        for doc in training_articles:
            counts.update(doc["text"].split())

        vocab_dict = {"[PAD]": 0, "[UNK]": 1}
        for word, _count in counts.most_common(max_tokens - 2):
            vocab_dict[word] = len(vocab_dict)
        return vocab_dict

    vocab = build_vocab(train, max_tokens=5000)
    return (vocab,)


@app.cell
def _(dev, label_to_id, test, torch, train, vocab):
    def encode_text(text):
        word_list = text.split()[:30]
        token_ids = [vocab.get(word, 1) for word in word_list]
        token_ids += [0] * (30 - len(token_ids))  
        return token_ids

    def to_tensors(subset):
        X = torch.tensor([encode_text(item["text"]) for item in subset], dtype=torch.long)
        Y = torch.tensor([label_to_id[item["label"]] for item in subset], dtype=torch.long)
        return X, Y

    X_train, Y_train = to_tensors(train)
    X_valid, Y_valid = to_tensors(dev)
    X_test, Y_test = to_tensors(test)

    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_valid:", X_valid.shape, "X_test:", X_test.shape)
    return X_test, X_train, X_valid, Y_test, Y_train, Y_valid


@app.cell
def _(nn, torch):
    class NewsClassifier(nn.Module):
        def __init__(self, vocab_size, num_classes, embedding_dim=200, dropout=0.5):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.blocks = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False)
                for _ in range(5)  
            ])
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(embedding_dim, num_classes, bias=False)

        def forward(self, x):
            x = self.embedding(x) 
            x = x.mean(dim=1)     
            for layer in self.blocks:
                x = self.dropout(x + torch.relu(layer(x)))
            return self.output(x)  

    return (NewsClassifier,)


@app.cell
def _(NewsClassifier, labels, vocab):
    model = NewsClassifier(vocab_size=len(vocab), num_classes=len(labels))
    print(model)
    return (model,)


@app.cell
def _(X_train, X_valid, Y_train, Y_valid, model, torch):
    MAX_EPOCHS = 60
    BATCH_SIZE = 32
    PATIENCE = 10
    START_MONITOR = 10

    def train_network():
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_accuracy = -1.0
        best_weights = None
        bad_epochs = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = torch.randperm(len(X_train))
            total_loss = 0.0
            total_correct = 0

            for batch_ids in order.split(BATCH_SIZE):
                xb, yb = X_train[batch_ids], Y_train[batch_ids]
                optimizer.zero_grad()         
                logits = model(xb)             
                loss = criterion(logits, yb)   
                loss.backward()               
                optimizer.step()              
                total_loss += loss.item() * len(batch_ids)
                total_correct += (logits.argmax(dim=1) == yb).sum().item()

            train_loss = total_loss / len(X_train)
            train_acc = total_correct / len(X_train)

            model.eval()
            with torch.no_grad():
                valid_logits = model(X_valid)
                val_loss = criterion(valid_logits, Y_valid).item()
                val_acc = (valid_logits.argmax(dim=1) == Y_valid).float().mean().item()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            elif epoch > START_MONITOR:
                bad_epochs += 1

            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch {epoch:3d} | train acc={train_acc:.3f} | val acc={val_acc:.3f}")

            if epoch > START_MONITOR and bad_epochs >= PATIENCE:
                print("EarlyStopping : arrêt anticipé car la précision sur l'ensemble de validation ne s'améliore plus.")
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)  
        return history

    history = train_network()
    return (history,)


@app.cell
def _(history, plt):
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 4))
    ax_accuracy.plot(history["train_acc"], label="Train accuracy")
    ax_accuracy.plot(history["val_acc"], label="Validation accuracy")
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.legend()
    ax_accuracy.grid(True, alpha=0.3)
    plt.show()
    return


@app.cell
def _(X_test, Y_test, classification_report, labels, model, torch):
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test)
        test_pred = test_logits.argmax(dim=1)
        test_loss = torch.nn.functional.cross_entropy(test_logits, Y_test).item()
        test_acc = (test_pred == Y_test).float().mean().item()

    print(f"Perte sur l'ensemble de test : {test_loss:.4f}")
    print(f"Précision sur l'ensemble de test : {test_acc:.2%}")

    print(classification_report(
        Y_test.numpy(),
        test_pred.numpy(),
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0
    ))
    return (test_pred,)


@app.cell
def _(ConfusionMatrixDisplay, Y_test, labels, plt, test_pred):
    fig_confusion, ax_confusion = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        Y_test.numpy(), test_pred.numpy(),
        labels=list(range(len(labels))), display_labels=labels,
        xticks_rotation=45, ax=ax_confusion,
    )
    plt.show()
    return


@app.cell
def _(model, plt, vocab):
    from sklearn.decomposition import PCA

    weights = model.embedding.weight.detach().cpu().numpy()
    token_items = list(vocab.items())[2:120] 
    word_ids = [idx for _word, idx in token_items]
    if len(word_ids) >= 2:
        projected = PCA(n_components=2).fit_transform(weights[word_ids])
        fig_emb, ax_emb = plt.subplots(figsize=(10, 6))
        ax_emb.scatter(projected[:, 0], projected[:, 1], alpha=0.5)
        for i in range(min(35, len(token_items))):
            ax_emb.annotate(token_items[i][0], projected[i], fontsize=7)
        ax_emb.set_title("Embedding (PCA visualization)")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
