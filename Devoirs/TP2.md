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


- Sur la taille du corpus et l'instabilité des résultats :
J'ai remarqué qu'en travaillant sur un très petit corpus (`load_data()[:200]`), les résultats entre PyTorch et Keras fluctuaient énormément et n'étaient pas les mêmes. J'ai compris que cela venait de la taille minuscule du jeu de test généré (à peine une quinzaine de phrases), où une seule bonne ou mauvaise prédiction fait varier le score de près de 8%.

Pour corriger cela et obtenir une comparaison fiable, j'ai choisi d'agrandir le corpus pour l'évaluation en passant à `load_data()[:2000]` et en réduisant le nombre d'itérations. Les deux modèles ont alors convergé de manière beaucoup plus stable vers un résultat d'environ 76,80% de précision.

- Suggestion sur l'ordre du sujet pour les prochains TPs :
Serait-il possible, pour les prochains devoirs, de placer la section "Réaliser une évaluation propre (division du corpus)" avant la construction et l'entraînement des modèles ?
J'ai suivi la liste des consignes dans l'ordre, et je ne pense pas être la seule à avoir codé et testé tous les réseaux sur l'ensemble du corpus, pour me rendre compte seulement à la fin qu'il fallait en fait diviser les données en amont (Train/Dev/Test) pour éviter que le modèle n'apprenne par cœur. Un ordre chronologique éviterait de devoir restructurer son code après coup.
