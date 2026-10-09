Sujet du 6 octobre, à pusher le lundi 12 octobre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 langues

- Modifier le chargement des données pour bien avoir 6 langues
- Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées)
- Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`)

- bonus (long): suivre le même principe en utilisant TextVectorizer de keras (on verra ça la semaine prochaine)

## Réaliser une évaluation propre 

- diviser correctement le corpus pour avoir des scores valides (j'ai decider de suivure le modèle vu en cours de m1 de 3 corpus 70/15/15)
- bonus: faire une matrice de confusion


## Expliquer ci-dessous vos difficultés ou posez vos questions

###Les difficultés et problème pricipales
- Le nom de variable qui ne peuvent pas être réutilisé
- Tentative de rester proche du code du tp1 donc peut etre pas le plus optimiser et prend beaucoup de temps dans la partit apprentissage (peu prendre jusqua 8 min)

####Chargement du corpus:
- Le chemin que j'avais dans ma première version ne fonctionnait pas car Marimo ne se lacer pas la ou était le fichier notebook et donc le corpus

####Separation du corpus
- J'ai suivis les tailles et formats conseillé en cours avec 3 corpus donc le plus gros pour l'entrainement
- J'ai choisis d'utiliser scikit-learn car il y a deja les outils pour split le corpus en sous corpus et ma tentative de le faire manuellement c'est avérer bien plus long et compliqué que d'utiliser ces outils
- J'ai essayer de répartir plus ou moin equitablement entre les langues du corpus avec option stratify, pour eviter que 1 langues soit plus representé dans l'entrainement

####Ecart type / moyenne
- J'ai calculé totu sur entrainement seulement pour garder 0 contact avec les sous corpus teste et verif
- J'ai eu une boucle infini lors de mes première tentative et que j'ai régler avec le v.append(val) et la reformulation de la boucle du dico

####Normalisation
- utilisation du z-score comme pour tp1
- J'ai utiliser l'ecart type et moyenne de entranemtn pour que modèle ne sache toujour rien des autres sous corpus

####Transformation en tableau
- Prépare les donnée pour sparse_categorical_crossentropy (J'ai encore du mal à savoir quand il est plus approprié d'utiliser les différentes option disponible)
- Doit préciser float32 pour x pour eviter des erreur mais int32 pour Y car c'est le format attendu par keras. (je n'ai pas encore bien compris la raison concrète)

####Apprentisage
- Pas de 0,1 ce qui donne un bon resultats mais peut etre raison de l'apprentissage tres long (8 min pour teste A, 7 pour le teste B)
- J'ai encore du mal à savoir quand il est plus approprié d'utiliser les différentes fonction possible entre tanh, softmax, sigmoid etc
- Garde meme strcuture que TP1 globalement sinon j'ai tendance à me perdre dans quoi mettre ou et quoi modifier
- Teste fait avec 2 autre version avec brespectivement plus de neuronne de de couche et tres peu. Les resultats sont similaire donc le nobres de neuronne ou de couche n'a pas l'aire d'avoir d'impact significatife sur la qualité des résultats.
