Sujet du 6 octobre, à pusher le lundi 12 octobre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 langues

- [x] Modifier le chargement des données pour bien avoir 6 langues ✅ 2026-10-11
	- déjà j'ai mis le fichier avec 6 langues
	- je me suis aussi amusée à rajouter plus de bigrammes (il y en a 7, donc maintenant la couche d'entrée a une taille de 7)
- Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées)
	- en keras : la dernière couche (sortie) a 6 noeuds, l'idée est qu'avec l'activation `softmax` derrière on se retrouve à avoir un vecteur à 6 dimensions qui comportent chacune une proba entre 0 et 1, la somme valant 1. Un coordonnée = une langue (proba que ce soit dans cette langue-là). J'ai agrandi un peu la couche dense intermédiaire du MLP (à 2 couches) à 10 noeuds.
	- en PyTorch : 
- Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`)

- bonus (long): suivre le même principe en utilisant TextVectorizer de keras (on verra ça la semaine prochaine)


## Réaliser une évaluation propre 

- diviser correctement le corpus pour avoir des scores valides
- bonus: faire une matrice de confusion


## Expliquer ci-dessous vos difficultés ou posez vos questions

(ou dites si tout va bien !)
...

