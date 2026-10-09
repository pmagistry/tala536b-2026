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

Aucune piste à proposer :-(

## Expliquer ci-dessous vos difficultés ou posez vos questions

J’ai trouvé le TP assez difficile mais très instructif. Ça m’a obligé à rechercher et à comprendre les notions de base que je ne connaissais pas du tout ! Je les ai étudiées sur internet (je n’ai pas encore eu le temps de me plonger dans le livre « Speech and Language Processing »). J’ai bien compris les principes, ainsi que les outils mathématiques.

En étudiant le code ligne par ligne, j’ai également bien assimilé sa structure et j’ai pu faire une partie des exercices en prenant comme exemple le code fourni. Mais je suis conscient qu’à l’heure actuelle, je ne serais pas capable d’écrire ce genre de code en partant de zéro.

Difficultés et questions :
- Je ne comprends pas bien pourquoi on utilise comme loss la fonction avec Logits (BCEWithLogitsLoss) et non sans. D’après ce que j’ai compris, le logit est le résultat brut avant l’application de la sigmoïde.
...
