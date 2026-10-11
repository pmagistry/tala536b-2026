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

## Difficultés rencontrées

- **Passer de 2 à 6 classes.** Changer le nombre de neurones de sortie ne suffisait pas : il fallait aussi encoder les labels en vecteurs one-hot, utiliser un softmax à la place de la sigmoïde et passer à `categorical_crossentropy`. Ces trois changements vont ensemble : avec 6 langues, le modèle donne une probabilité par langue, donc les labels et la loss doivent avoir le même format.

**Des features peu adaptées à 6 langues.** Les fréquences de "th" et de "en" avaient été choisie pour distinguer l'anglais de l'allemand. Avec 6 langues, le modèle atteint à pe près 52% de précision sur le test, avec des scores train et test très proches. Je pense que la limite vient des features et de l'absence de couche cachée plutôt que de l'entraînement, mais je n'ai pas encore testé d'autres bigrammes ni un MLP pour le vérifier.

