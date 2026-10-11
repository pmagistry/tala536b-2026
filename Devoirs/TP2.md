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

Bonsoir, je pense que dans l'ensemble cela ne s'est pas si mal passé de mon côté. J'ai surtout eu du mal à comprendre quoi faire exactement. Ayant manqué le dernier cours avec les manifestations c'était pour moi un peu touffu. J'ai donc dû me renseigner pour comprendre quoi modifier. J'ai choisit 6 features qui collaient plus au moins à une langue différentes, gardant l'idéal originale pour le training. J'ai eu des problèmes notamment au niveau des dimensions à l'initialisation des modèles (la partie suivante après modification: keras.layers.Input(shape=(X6_lang.shape[1],))) où je ne comprenais pas d'où venait l'erreur et j'ai demandé à claude au bout d'une bonne heure pour qu'il me donne cette solution et me l'explique. Avant j'essayais juste de modifier à la main en mettant un autre nombre que 2 ou 3, mais ça plantait directement en disant que j'avais 32 input. Je me retrouve donc avec environ 80% d'acc pour les deux modèles.

