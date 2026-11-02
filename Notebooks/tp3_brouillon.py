import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import utils_tp2 as u

# Lecture des datas
corpus = u.load_text("./corpus6.txt", limit=100000)   # limit assez grand pour tout lire
langues = list(sorted({d['label'] for d in corpus}))
code_langues = {l: i for i, l in enumerate(langues)}

train, test = train_test_split(corpus, train_size=0.8, random_state=42)
train, valid = train_test_split(train, train_size=0.8, random_state=42)

Y_train = np.array([code_langues[d['label']] for d in train])
Y_valid = np.array([code_langues[d['label']] for d in valid])
Y_test  = np.array([code_langues[d['label']] for d in test])

# Hyperparamètres
MAX_TOKENS = 5000     # taille du vocabulaire (index 0 = padding, 1 = [UNK])
SEQ_LEN = 200         # longueur fixe des séquences (tronquées / complétées par des 0)
EMBED_DIM = 32        # dimension des plongements
DROPOUT = 0.3         # hyperparamètre à donner

# TextVectorization en mode "int"
# Sortie : (batch, SEQ_LEN) d'entiers, un entier par token
tv = keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=None,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
)
tv.adapt([line["text"] for line in train])   # adapt sur le train uniquement

X_train = tv([line["text"] for line in train])
X_valid = tv([line["text"] for line in valid])
X_test = tv([line["text"] for line in test])
# Y_* : encodage sparse (entiers), comme dans TP2


# Modèle
def model_embedding(nb_classes, pooling="avg"):
    pool = (layers.GlobalAveragePooling1D() if pooling == "avg"
            else layers.GlobalMaxPooling1D())
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN,), dtype="int32"),
        # (batch, SEQ_LEN) -> (batch, SEQ_LEN, EMBED_DIM)
        layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM, mask_zero=True),
        # (batch, SEQ_LEN, EMBED_DIM) -> (batch, EMBED_DIM)
        pool,
        layers.Dropout(DROPOUT),
        layers.Dense(64, activation="relu"),
        layers.Dense(nb_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",          # converge bien plus vite que SGD ici
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = model_embedding(len(labels), pooling="avg")
model.summary()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=30, batch_size=32,
)
model.evaluate(X_test, Y_test)

# Courbes train / validation pour repérer l'overfitting
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="valid")
plt.legend(); plt.show()
