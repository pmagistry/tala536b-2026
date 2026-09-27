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

...

## Expliquer ci-dessous vos difficultés ou posez vos questions

- Il fallait rétrograder le module numpy (<2.0) car la version la plus récente de torch que mon pc supporte est 2.2.2.  
- Il manque polars dans les modules prérequis
- Je dois aussi downgrade la version de keras (3.6.0) afin d'éviter l'erreur "flatten_with_keys_fn" à cause de comptabilité.
- Ce sera super si l'endroit de données requises est précisé à l'avance
- Un problème dû à MPS (NotImplementedError: The operator 'aten::\_foreach_add_.List' is not currently implemented for the MPS device) et os.environ\["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" n'a pas d'effet. 
- Au final il fallait indiquer que MPS n'est pas dispo sur la machine afin de passer toutes les opérations sur CPU.
