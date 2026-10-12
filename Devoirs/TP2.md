Sujet du 6 octobre, à pusher le lundi 12 octobre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 langues

- Modifier le chargement des données pour bien avoir 6 langues
- Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées)
- Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`)

- bonus (long): suivre le même principe en utilisant TextVectorizer de keras (on verra ça la semaine prochaine)


## Réaliser une évaluation propre 

- diviser correctement le corpus pour avoir des scores valides
- bonus: faire une matrice de confusion


## Expliquer ci-dessous vos difficultés ou posez vos questions

J'ai passé beaucoup de temps à comprendre le code du notebook, notamment le fonctionnement du réseau de neurones et les différentes étapes de l'entraînement.

Pour cet exercice, j'ai adapté le chargement des données pour prendre en compte les six langues du corpus. J'ai également modifié l'encodage des étiquettes, la structure du réseau avec une sortie en softmax, ainsi que la fonction de perte pour la classification multiclasse. Enfin, j'ai divisé les données en ensembles d'entraînement et de test pour mieux évaluer le modèle.

Ces modifications m'ont demandé beaucoup de temps, je n'ai donc pas eu suffisamment de temps pour réaliser les exercices bonus.
