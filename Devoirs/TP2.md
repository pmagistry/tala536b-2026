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

J'ai notamment eu un problème avec la forme ciblé et la sortie. Voici l'erreur que j'ai eu :
```
Arguments `target` and `output` must have the same shape. Received: target.shape=torch.Size([32]), output.shape=torch.Size([32, 6])
```
Pour regler cette erreur, j'ai modifier la loss function et suis passé de categorical_crossentropy. à sparse_categorical_crossentropy car cette fonction sait faire la comparaison de différentes classes d'elle-même.

Je suis aussi passé de 10 epochs à 25 pour permettre à l'algorithme de mieux apprendre.
