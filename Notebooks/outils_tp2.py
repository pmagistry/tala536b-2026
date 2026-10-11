import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Liste unique des séquences de lettres utilisées comme entrées du modèle
PATTERNS = ["th", "en", "es", "ij", "aa", "ll", "ñ", "ou", "zz", "cc"]


def load_data_6_languages(filepath="../corpus6.txt", patterns=PATTERNS):
    """
    Lit le corpus (une ligne = 'langue texte') et calcule, pour chaque
    séquence de lettres, sa fréquence relative dans le texte.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            label, text = parts[0], parts[1]
            length = len(text) if len(text) > 0 else 1
            features = {pat: text.lower().count(pat) / length for pat in patterns}
            features["label"] = label
            data.append(features)
    return data


def prepare_features_and_labels(dataset, patterns=PATTERNS):
    """
    Construit la matrice X (une colonne par séquence) et encode les langues
    en entiers 0 à 5. La normalisation se fait ensuite, dans le notebook,
    après la coupe train/test.
    """
    X = np.array([[d[p] for p in patterns] for d in dataset])
    labels_raw = [d["label"] for d in dataset]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_raw)
    return X, y, encoder


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Trace la matrice de confusion sous forme de carte de chaleur.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Classes prédites")
    ax.set_ylabel("Classes réelles")
    ax.set_title("Matrice de Confusion")
    return fig
