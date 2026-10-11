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

J'ai encore des doutes sur comment on défini le nombre de 'units' lors de la déclaration des couches, principalement les couches cachées. Pour ce qui est des couches d'entrées et de sortie, j'ai compris que la taille des units dépendent des dimensions entrantes et sortantes. Pour cette fois, dans la couche cachée du modèle de reconnaissance des 6 langues, j'ai choisi de mettre une taille de 64, car, en faisant des recherches sur le type d'activation tanh, j'ai vu que les puissances de 2 sont souvent utilisées, j'ai donc testé 8,16 et 32, et ai fini par prendre 64, qui me donnait les meilleures résultats, mais je ne suis pas sûre de comprendre comment le savoir d'emblée, et ce que cela représente réellement.


...
