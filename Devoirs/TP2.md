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

J'ai beaucoup expérimenté en variant le nombre de neurones parallèles, en ajoutant des couches cachées en plus, en changeant la fonction d'activation. J'ai encore du mal à comprendre ce qui concrètement me motiverait d'opter pour une fonction plutôt qu'une autre. J'imagine qu'il y a un compromis entre information et complexité, avec certaines fonctions moins coûteuses mais qui nullifient certaines différences.

J'en ai profité pour faire varier la taille du corpus d'entrainement, je me suis arrêté à 2000 lignes en optant pour 10% du corpus dédié à l'évaluation. Pour le découpage du corpus j'ai utilisé le splicing de python, je me doute qu'il y a mieux mais c'est ce qui m'a semblé intuitif.

Keras m'a semblé assez intuitif à utiliser, avec quasiment rien à modifier. J'ai eu beaucoup plus du mal avec pytorch car je sens que je n'ai pas encore bien compris comment la librairie fonctionne. J'ai utilisé l'ia pour comprendre comment faire, j'avais en fait plusieurs problèmes : j'avais un softmax en fin de mon nn.sequential alors que CrossEntropyLoss intègre déjà un softmax, aussi je n'avais pas compris quel type de donnée cette fonction exigeait (tenseur 1D et non un tenseur de vecteurs one hot). Je sens que je suis encore très fravile là-dessus, même l

J'ai un peu buté sur comment faire pour adapter le gold pour 6 langues. Ou plutôt j'ai compris ce qu'il fallait faire et j'ai su le faire avec les array de numpy mais j'ai eu du mal avec les tensor de torch.

Je me suis très brièvement penché sur TextVectorizer mais je n'ai pas essayé de l'implémenter. J'ai par contre fait la matrice de confusion, en tatonnant pas mal car j'ai mis du temps à comprendre qu'on devait utiliser model.predict() pour keras. C'était assez satisfaisant de rendre interprétable les distributions de probabilités en sortie du modèle, pour le plot j'ai repris l'exemple type de la documentation.
