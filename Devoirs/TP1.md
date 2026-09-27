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

Pour classer les documents on pourrait :
- utilisé scikit-learn car il permet déjà de séparer par domaine.
- faire un dictionnaire de langue, avec des mots les plus courants dans les langues, un peu comme les stopwords.

## Expliquer ci-dessous vos difficultés ou posez vos questions

J'ai eu quelques problèmes pour charger mon environnement et keras car il cherchait un "tree" que je n'avais pas, j'ai donc ajouté dans la liste des programmes à installer `dm-tree`.

Pour comprendre le code, j'ai du le reprendre à la main pour comprendre les différentes fonctions, pourquoi elles étaient utilisées. Néanmoins, je ne suis pas sure de bien avoir compris comment les utiliser pour reprendre le code en pytorch.
