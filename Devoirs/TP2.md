Sujet du 6 octobre, à pusher le lundi 12 octobre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 langues

- Modifier le chargement des données pour bien avoir 6 langues ☑️
- Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées) ☑️
- Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`) ☑️

- bonus (long): suivre le même principe en utilisant TextVectorizer de keras (on verra ça la semaine prochaine)


## Réaliser une évaluation propre 

- diviser correctement le corpus pour avoir des scores valides ☑️
- bonus: faire une matrice de confusion


## Expliquer ci-dessous vos difficultés ou posez vos questions

* Je n'ai pas bien compris ce que X_lang attendait comme donnée, donc j'ai au départ donnée les labels des langues. 
* En raison d'un nombre d'epoch trop peu, l'accurracy avec Keras est largement plus basse que le teste fait avec Torch. Torch tourne plus vite, j'ai changé le nombre d'epoch du modèle de keras pour qu'il soit égal à torch, l'accurracy augmente à partir des 50 epochs et arrive à atteindre un taux similaire de torch à partir de 500 epoch. Mais le script tourne plus beaucoup plus longtemps.
* Avec keras, la division n'était pas obligatoire puisqu'il était déjà fait avec `validation_split`. Cependant, j'ai tout de même testé la division manuel et je remarque que le résultat est largement meilleur.