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

## Aprés avoir fait le tp1, je me suis familiarisé avec certaines notions et fonctions de keras et pytorch.En commençant le tp2 je me suis rendu compte que pour travailler sur plusieurs langues(plus de 2) il faudra utiliser dautrs fonctions qui peuvent traiter plusieurs entrées et sorties. je parle ici de la fonction softmax par exemple qui transforme les scores en probabilités de sorte que la somme de toutes ces probailités vaut 1
## j'etais contraint aussi de changer la fonction BCE utilisée dans le modele avec 2 langues par categorical Cross Entropy qui permet de traiter plusieurs langues.
## Concernant la couche utilisée, j'ai conservé le 8 neurones comme c'etait le cas dans le tp1.

## J'ai pas réussi à faire la meme chose avec pytorch puisque ce dernier n'utilise pas les memes fonctions 
