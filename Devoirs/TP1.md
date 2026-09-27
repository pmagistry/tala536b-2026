# Exercices du 22 septembre (à pusher le dimanche 27)

## Environnement de travail.

- cloner le github
- se créer un environnement virtuel avec `uv`, installer les bibliothèques torch, keras, marimo, matplotlib, scikit-learn.
- essayer de lancer et de faire tourner le notebook de la première séance (`marimo edit cours1.py`)

## Lecture et modification du code

- s'assurer que l'on comprend le code du notebook
- essayer de reprendre le code en pytorch pour effectuer la classification de langues donnée en keras.
- essayer d'ajouter une couche au réseau défini en Keras
- essayer d'ajouter un troisième bigramme de lettre en entrée du réseau.

## Proposer des pistes (sans coder) pour classer des documents entre 3 langues ou plus.

- compter les bigrammes les plus et les moins fréquents pour chaque langue dans de larges corpus et construire un NN avec chaque bigramme en input
- entrainer un modele par apprentissage non supervisé à reconnaître des features redondants dans chaque langue
- s'inspirer des règles de typologie linguistique pour récupérer les features les plus intéressants en terme de performance et de complexité
- effectuer une analyse syntaxique automatique et l'utiliser comme features dans le NN

## Expliquer ci-dessous vos difficultés ou posez vos questions

- Pour l'installation, tout va bien sauf qu'il manquait _polars_ dans la liste des bibliothèques requises
- un fichier de type _requirements.txt_ pour l'option -r de `pip install` serait plus adapté

- sans modifier, le modèle NN donné en keras ne donne pas toujours une bonne précision, Je me dis que c'est peut-être le learning rate qui est trop grand et qui empêche de s'approcher doucement d'un coût minimal. J'ai donc modifié le param learning_rate de l'optimizer SGD. J'ai mis 0.1 et ça marche très bien alors que la valeur par défaut est 0.01. Sa valeur était en fait trop petite, ce qui peut aussi empêcher de s'approcher d'un coût minimal si on part de trop loin

- dans l'implémentation pytorch, j'ai essayé de mettre un learning rate très grand comme 10 ou 20 et ça marche très bien, mais je pense que c'est simplement parce que la distinction th et en pour eng et deu marche très bien. Pas sûr que ça fonctionne bien pour 3 langues

- si on ajoute une couche au réseau, je la mets intuitivement avant la couche déjà présente qui a la fonction d'activation sigmoid pour transformer l'espace puis avoir un perceptron final qui représente une fonction affine. Je ne suis pas sûr du nombre optimal de neurones donc je mets un peu au hasard. Les résultats sont à peu près les mêmes quel que soit le nombre de neurones. Je ne sais pas quoi mettre comme fonction d'activation donc je laisse activation linéaire par défaut. J'ai aussi essayé relu pour apprendre ce que c'est.

### Ce que j'ai retenu

- j'ai appris la notation 1{...} pour la fonction indicatrice
- je remarque que pour entraîner le premier perceptron, sans changer les paramètres, le nombre d'epochs nécessaires pour arriver à une itération sans aucune erreur peut être drastiquement différent à chaque fois. Ça peut aller de 1 à 100.
- j'avoue que je n'ai pas creusé la question de pourquoi utiliser la fonction BCEWithLogitsLoss comme fonction Coût mais j'aimerais bien en apprendre plus
- à retenir : with torch.no_grad():
- la normalisation permet un lissage des donnés. En calculant la cote Z de chaque proportion d'apparition d'un bigramme de lettre, on évite des valeurs trop extrêmes et la valeur 0.
