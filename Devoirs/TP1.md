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

L'année dernière, lors du cours Traitement Statistique des Données, la classification en plus de 2 classes avait été evoquée. On pourrait donc soit recourir à plusieurs classifieurs bianaires, qui pour chaque classe, déterminent si un document lui appartient, ou à un seul classifieur multiclasses. Pour implémenter un perceptron multiclasses, on pourrait essayer de changer le type d'activation, de sigmoïde vers autre chose.

## Expliquer ci-dessous vos difficultés ou posez vos questions

Pour l'instant, nous n'avons pas encore vu la programmation orientée objet, donc certains aspects du code peuvent être un peu flous.

Pour ajouter une couche dans keras, j'ai eu beaucoup de difficultés, même en allant consulter la documentation de la bibliothèque. J'ai fini par rajouter un nouvel appel à la fonction `Dense()` dans la déclaration du réseaux avec `Sequential()`, cette solution étant la seule qui ne renvoie pas un message d'erreur, mais je ne suis pas sûre que ce soit ce qui est demandé.




