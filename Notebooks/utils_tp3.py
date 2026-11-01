


            if limit is not None and len(data) >= limit:
                break
    return data


def normalise_data(dataset):
    for k in dataset[0].keys():
        if k != "label":
            mean = np.mean([d[k] for d in dataset])
            std = np.std([d[k] for d in dataset])
            if std == 0:
                std = 1
            for d in dataset:
                d[k] = (d[k] - mean) / std


def encode_corpus_one_hot(dataset: list[dict], langues: list[str]):
    bigrams_keys = list(sorted(k for k in dataset[0].keys() if k != "label"))
    X_lang = np.array([[d[k] for k in bigrams_keys] for d in dataset])
    y_lang = np.array([[1 if d["label"] == l else 0 for l in langues] for d in dataset])
    return X_lang, y_lang


def encode_corpus_sparse(dataset: list[dict], langues: list[str]):
    bigrams_keys = list(sorted(k for k in dataset[0].keys() if k != "label"))
    X_lang = np.array([[d[k] for k in bigrams_keys] for d in dataset])
    code_langues = {l: i for i, l in enumerate(langues)}
    y_lang = np.array([code_langues[d["label"]] for d in dataset])
    return X_lang, y_lang


def model_sparse(nb_input=2, nb_classes=6, optim="sgd"):
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        Normalization(),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(nb_classes, use_bias=False, activation="softmax")
    ])
    model.compile(optimizer=optim, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def model_one_hot(nb_input=2, nb_classes=6, optim="sgd"):
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        Normalization(),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(nb_classes, use_bias=False, activation="softmax")
    ])
    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def model_embedding(nb_input, nb_classes, vocab_size, pooling="max", optim="adam"):
    pooling_layer = GlobalMaxPooling1D() if pooling == "max" else GlobalAveragePooling1D()
    model = Sequential([
        keras.layers.Input(shape=(nb_input,)),
        Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
        pooling_layer,
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(nb_classes, activation="softmax")
    ])
    model.compile(optimizer=optim, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

