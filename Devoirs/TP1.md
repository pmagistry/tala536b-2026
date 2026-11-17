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
Pour reconnaître trois langues ou davantage, on pourrait commencer par constituer un corpus contenant des exemples de chaque langue.

On pourrait ensuite utiliser plusieurs bigrammes caractéristiques de ces langues, plutôt que de se limiter à th, en et ch.

Il faudrait également adapter la couche de sortie du réseau. Au lieu d'un seul neurone pour distinguer deux classes, on pourrait utiliser un neurone par langue, avec une fonction softmax pour obtenir une distribution de probabilités sur les différentes langues.

Enfin, il serait intéressant de tester le modèle sur des textes qui n'ont pas servi à son entraînement, afin de vérifier s'il arrive à reconnaître de nouveaux exemples.

## Expliquer ci-dessous vos difficultés ou posez vos questions

J'ai eu des difficultés à comprendre comment passer du modèle Keras à son équivalent PyTorch, notamment pour la conversion des données en tenseurs, la fonction de coût et les différentes étapes de l'entraînement.

J'ai également rencontré des erreurs lorsque j'ai modifié les dimensions d'entrée du réseau. Par exemple, après l'ajout du troisième bigramme, il fallait adapter le modèle pour qu'il accepte trois caractéristiques.

J'ai dû faire plusieurs essais avant d'obtenir un code qui s'exécute correctement.

Ce TP m'a surtout permis de mieux comprendre le rôle des données d'entrée, des couches du réseau, de la fonction d'activation et de l'entraînement. Je dois encore approfondir le fonctionnement de la rétropropagation et la manière dont le réseau ajuste ses poids.

### Résultats des expériences

**1. Classification en PyTorch**

J'ai repris la classification anglais/allemand avec les deux bigrammes `th` et `en`. J'ai utilisé un modèle `nn.Linear(2, 1)`, l'optimiseur SGD avec un taux d'apprentissage de 0.1 et la fonction de coût `BCEWithLogitsLoss`.

Après 500 itérations, j'ai obtenu une précision de **97 %** sur les 200 exemples utilisés.

**2. Ajout d'une couche cachée en Keras**

J'ai ajouté une couche de 8 neurones avec la fonction d'activation `relu`, avant la couche de sortie `sigmoid`.

Avec 10 époques, j'ai obtenu une précision de **85,5 %**. En passant à 50 époques, j'ai obtenu **95,5 %**.

**3. Ajout d'un troisième bigramme**

J'ai choisi le bigramme `ch`, en complément de `th` et `en`. J'ai modifié la fonction `load_data()` et adapté la dimension d'entrée du modèle Keras pour prendre en compte les trois caractéristiques.

Avec 50 époques, j'ai obtenu une précision de **96,5 %**.

Les précisions affichées ont été calculées sur les 200 exemples chargés, dont une partie a également servi à la validation du modèle Keras. Il ne s'agit donc pas d'une évaluation sur un jeu de test indépendant.
