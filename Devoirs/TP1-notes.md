# Exercices du 22 septembre (à pusher le dimanche 27)

## Environnement de travail.

- cloner le github
- se créer un environnement virtuel avec `uv`, installer les bibliothèques torch, keras, marimo, matplotlib, scikit-learn.


## Lecture et modification du code

- s'assurer que l'on comprend le code du notebook
- essayer de reprendre le code en pytorch pour effectuer la classification de langues donnée en keras.
- essayer d'ajouter une couche au réseau défini en Keras
- essayer d'ajouter un troisième bigramme de lettre en entrée du réseau.



## Proposer des pistes (sans coder) pour classer des documents entre 3 langues ou plus.

Je pense que pour pytorche il faut juste changer la partie lang_model = nn.Linear(3,1) en lang_model = nn.Linear(*nb de bigrammes*,***nb de langue du corpus***) ou plus selon le nombre de langue que l'on a pour avoir le nombre de sortie adapté.
Et pour keras je pense qu'il faut juste changer  loss='binary_crossentropy' en un mode pour plus de langue. En cherchant j'ai trouver les modes *sparse_categorical_crossentropy* et *categorical_crossentropy*.

Dans les deux cas je pense qu'il faut aussi jouer avec et adapté les paramettres de la perte pour que ce soit adapté au multiclassages.

## Expliquer ci-dessous vos difficultés ou posez vos questions

### Problèmes et solution rencontré pour la partie environement virtuel et github:  (nom de l'environement virtuel -> rnn)

- Lors de la création et du push initial de ma branche j'avais un problème d'identification, le mot de passe de mon compte github n'était pas accepter.  La solution à était de créer un Personal Access Token (PAT) sur github et l'utiliser entant que mot de passe avec mon identifiant (il est valable 90 apres il faudra le renouveler).
- Il a fallut ajouter tensorflow et polars à l'environement car sinon il y avait une problème avec keras lors des imports.


### Problèmes, solution et questions soulevés pour la partie notebook:

- Actuellement j'utilise le corpus 2 langues car je n'ai pas vu de différence particuliere quand on mettait le corpus 6 langues. (il faut changer des paramettre dans la partie keras et la partie pytorch pour s'adapter au fait qu'il y est plus de langues).
- Pour pytorch il a fallut ajouter dtype float32 sinon un erreur scalar apparaissait.De même il a fallut ajouter .unsqueeze(1) à y_lang pour qu'il soit sur le même plan que X.
- Pour changer en ajoutant une autre couche au modele de keras je n'ai pas vu de différences particulière entre le nombre de neuronne actif que je lui donné (teste fait avec 2, 3, 10).
- Pour le choix des bigrammes j'ai pris er ou ch parce que il y a l'air d'en avoir bcp en fr et allmand pour le corpus 2 langues (er actuellement implenter dans le programme).

- Je remarque que pour le modèle keras et le modèel pytorch, quand la cellule est relancer les resultats change énormement, que ce soit la précision, les poids ou le biais. Je pense que cela à avoir ave l'aléatoire mais je n'ais pas trouvé comment regler ce problèmes.
