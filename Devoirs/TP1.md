# Exercices du 22 septembre (à pusher le dimanche 27)

## Environnement de travail.

- cloner le github
- se créer un environnement virtuel avec `uv`, installer les bibliothèques torch, keras, marimo, matplotlib, scikit-learn.
- essayer de lancer et de faire tourner le notebook de la première séance (`marimo edit cours1.py`)

## Lecture et modification du code

- s'assurer que l'on comprend le code du notebook +
- essayer de reprendre le code en pytorch pour effectuer la classification de langues donnée en keras. +
- essayer d'ajouter une couche au réseau défini en Keras
- essayer d'ajouter un troisième bigramme de lettre en entrée du réseau. 

## Proposer des pistes (sans coder) pour classer des documents entre 3 langues ou plus.

je ne sais pas ...

## Expliquer ci-dessous vos difficultés ou posez vos questions

- Normalisation: j'ai mis du temps à comprendre les 55% en absence de normalisation; j'attriburais d'abord ça au vanishing gradient mais ensuite même avec une multiplication des epochs le résultat ne s'améliorait pas. dans ce cas là je crois que seul le biais apprend.

- Boucle d'entraienement pour le mini-batch: j'avais d'abord placé zero_grad() et step() en dehors de la boucle 

----

IA utilisé pour le shuffle du mini-batch et pour l'utilisation de Dataset plutot que de le faire à la main dans la partie réécriture de keras en pytorch 