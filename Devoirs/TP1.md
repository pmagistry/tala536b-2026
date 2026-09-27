# Exercices du 22 septembre (à pusher le dimanche 27)

## Environnement de travail.

- cloner le github
- se créer un environnement virtuel avec `uv`, installer les bibliothèques torch, keras, marimo, matplotlib, scikit-learn.
- essayer de lancer et de faire tourner le notebook de la première séance (`marimo edit cours1.py`)

## Lecture et modification du code

- s'assurer que l'on comprend le code du notebook
- essayer de reprendre le code en pytorch pour effectuer la classification de langues donnée en keras.
    j'y ai passé l'essentiel de mon temps et je n'ai pas vraiment l'impression d'avoir plus compris qu'au départ
- essayer d'ajouter une couche au réseau défini en Keras
    je crois avoir rajouté une couche mais je n'en suis même pas sûr honnêtement
- essayer d'ajouter un troisième bigramme de lettre en entrée du réseau.
    c'était très rapide à faire

## Proposer des pistes (sans coder) pour classer des documents entre 3 langues ou plus.

Sans trop être certain, je sens qu'il faudra à minima changer de fonction d'activation. Si le sigmoid est pratique avec 2 classes, rajouter une 3e classe nous empêche de l'utiliser. Du moins je crois.
Et aussi, et je pense que ce sera plus simple, il faut adapter le code pour pouvoir prendre une entrée

## Expliquer ci-dessous vos difficultés ou posez vos questions

J'ai pris un assez long temps pour me familiariser avec la documentation de pytorch et keras. J'avais déjà utilisé pytorch pour un projet l'année dernière mais c'était avec les options de base et je n'avais pas pris le temps de comprendre réellement comment ça fonctionne.

Même en essayant de faire un va et vient entre la documentation et l'exécution du code ligne par ligne j'ai du mal à comprendre comment l'entrainement fonctionne. Et surtout, j'ai du mal à comprendre quel type de donnée je manipule à chaque étape : est-ce que c'est un tensor ? oui, mais dans ce cas là c'est des float32 ou des float64. il faut squeeze ou il faut unsqueeze ? (j'ai pas bien compris non plus ce que ça fait exactement)

Quelle est la différence concrète entre un np.array et un torch.tensor ? Je vois qu'on utilise l'un pour keras et l'autre pour pytorch pur mais je ne sais pas l'expliquer.
