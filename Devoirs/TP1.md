# Exercices du 22 septembre (à pusher le dimanche 27)

## Environnement de travail.

- cloner le github
- se créer un environnement virtuel avec `uv`, installer les bibliothèques torch, keras, marimo, matplotlib, scikit-learn.
- essayer de lancer et de faire tourner le notebook de la première séance (`marimo edit cours1.py`)

## Lecture et modification du code

- s'assurer que l'on comprend le code du notebook
- essayer de reprendre le code en pytorch pour effectuer la classification de langues donnée en keras.
- essayer d'ajouter une couche au réseau défini en Keras
- essayer d'ajouter un troisième bigramme de lettre en entrée du réseau.

## Proposer des pistes (sans coder) pour classer des documents entre 3 langues ou plus.

...

## Expliquer ci-dessous vos difficultés ou posez vos questions

J'ai eu quelques difficultés au début avec l'environnement Python, notamment pour faire fonctionner Keras avec le backend PyTorch et pour installer certaines dépendances comme polars.

J'ai également eu besoin de clarifier la différence entre les logits et les probabilités, ainsi que le rôle de BCEWithLogitsLoss en PyTorch par rapport à l'utilisation d'une activation sigmoid avec binary_crossentropy dans Keras.

Après avoir repris le code étape par étape, la logique de l'entraînement et la correspondance entre Keras et PyTorch sont plus claires.
