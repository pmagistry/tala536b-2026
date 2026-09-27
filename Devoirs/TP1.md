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

Pour pouvoir classer des documents dans plus de 2 langues, il faut passer d'une classification binaire à une classification multiclasse. Le modèle pourrait produire une probabilité pour chaque langue, en se basant sur différentes caractéristiques, comme les bigrammes vus ici. Et la classe avec la probabilité la plus élevée serait alors la langue prédite.

## Expliquer ci-dessous vos difficultés ou posez vos questions

J'ai trouvé ça compliqué de lire et comprendre du code dans une librairie (Keras) que je ne connaissais pas, et ensuite de devoir l'adapter dans une autre librairie (Pytorch) que je ne connaissais pas.  
Je ne suis pas sûre d'avoir bien saisi la pertinence de l'ajout de nouvelles couches. J'ai compris que l'ajout d'une couche cachée avec une fonction d'activation non linéaire permet au réseau d'apprendre des relations plus complexes entre les caractéristiques, mais j'ai encore du mal à comprendre concrètement dans quels cas cela est nécessaire ou utile.
