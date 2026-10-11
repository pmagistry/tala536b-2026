Sujet du 6 octobre, à pusher le lundi 12 octobre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 langues

- [x] Modifier le chargement des données pour bien avoir 6 langues ✅ 2026-10-11
	- déjà j'ai mis le fichier avec 6 langues
	- je me suis aussi amusée à rajouter plus de bigrammes (il y en a 7, donc maintenant la couche d'entrée a une taille de 7)
- Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées)
	- **en keras** : la dernière couche (sortie) a 6 noeuds, l'idée est qu'avec l'activation `softmax` derrière on se retrouve à avoir un vecteur à 6 dimensions qui comportent chacune une proba entre 0 et 1, la somme valant 1. Un coordonnée = une langue (proba que ce soit dans cette langue-là). J'ai agrandi un peu la couche dense intermédiaire du MLP (à 2 couches) à 10 noeuds.
	- **en PyTorch :** **à faire...** 
- Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`)
	- **en Keras** : j'utilise la loss `categorical_crossentropy`, qui permet avec `softmax` de prédire un vecteur de distribution de probabilités pour chaque langue (vecteur de taille 6 comme on a 6 langues possibles). J'adapte en conséquence le vecteur "gold" `y_lang`, valeur de référence pour l'évaluation (et le calcul de la loss en fait ?? donc pour l'optimisation) qui devient un array contenant des vecteurs de taille 6.
	- **en PyTorch** : il n'y a pas d'équivalent direct à la loss `categorical_crossentropy`. Deux options : recoder manuellement une fonction de loss équivalente (comme proposé [ici](https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/4)), ou bien utiliser la loss cousine `CrossEntropyLoss` de la librairie PyTorch, qui fonctionne différemment en adaptant la valeur de référence `y_lang` au format de sortie avec cette loss. 
		- N'étant pas trop sûre de quoi faire, j'ai laissé ça de côté dans un premier temps...
- bonus (long): suivre le même principe en utilisant TextVectorizer de keras (on verra ça la semaine prochaine)


## Réaliser une évaluation propre 

- [x] faire marcher l'évaluation ✅ 2026-10-11
- diviser correctement le corpus pour avoir des scores valides
	- [ ] a priori : il faut entraîner le modèle sur une partie, évaluer ensuite sur une autre (cf train_test_split tout ça)
- bonus: faire une matrice de confusion


## Expliquer ci-dessous vos difficultés ou posez vos questions

- comment marche l'activation `softmax` ? 🤔
- différences `categorical_crossentropy` (Keras) et `CrossEntropyLoss` (PyTorch) ?