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

## Expliquer ci-dessous vos difficultés ou posez vos question
## Pour ce premier TP jereconnait que la tache était dur, j'ai mis beaucoup de temps pour comprendre le code et les différentes librairies utilisées. Au debut j'ai commencé par la documentation des deux (Keras et Pytorch). Ensuite j'ai essayé de faire une petite comparaison et sur leur mode de fonctionnenement.
## les divergences que j'ai réussi à voir par exemeple au niveau de la préparartion des données ou Keras utilise Numpy alors que pytorch travaille avec les tenseurs 
## Avec Keras on utilise Segmoid au tout debut contrairement à pytorch qui l'intègre dans le calcul de la perte   
