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

Pour classer des documents multilingue on peu ajuster le nombre de dimension en entrée et en sortie dans le code du reseau de neuronne en Keras par exemple s

## Expliquer ci-dessous vos difficultés ou posez vos questions

La comprehension du script n'a pas ete simple. 
Mais si j'ai bien compris les diffents script sont une sorte d'evolution du reseau de neurone de sa version simple a sa version "complexe" pouvant traiter des données non lineaire à l'aide du Stochastic gradient descent en incluant une couche cache  en keras et en pythorch. 
J'ai trouvé que la version pythorche etait plus parametrable et moins  abstraite tandis qu'en keras il y a moins de parametres mais c'est plus abstrait? 

Ce que j'ai eu du mal a comprendre : la decorraltion entre les features et les dimensions de sortie.
Est ce qu'on peu revoir le concept des classe et des method de la classe MLP?

        '''
        def forward(self, x):
        return self.out(torch.tanh(self.hidden(x)))
        
        '''

Pourquoi choisir 'tanh' plutot que 'sigmoid'? : images -1 à 1 tanh et 0ou 1 sigmoid ?

        '''
        
        Dense(1,activation='tanh'), #hidden layer de dimension 2,2 
        
        '''
## Recaptitulatif de la correction du tp:

J'ai repris le script cours1 et je l'ai comparé avece qui a ete fait en correction.

Je note que j'ai confondu beaucoup de notion comme la fonction d'activation et la metrique. 
J'ai compris qu'une fonciton d'activation (sigmoid , tanh etc ) et une metrique n'etaient pas la meme chose.
Les metriques sont plutot la precisions, le rappel ou la f-mesure.

Je pensais aussi que l'activation etait en quelque sorte le gradient, mais, 
en realité l'activation transforme les valeurs à l'aller (forward, je la choisis). 
Le gradient indique comment corriger les poids au retour (backward, calculé automatiquement).
Le gradient est multiplié par la pente de l'activation qu'il traverse.

Je pensais aussi que le gradient  agissait à chaque epoch.
Alors qu'il agit à chaque pas (batch).
Un epoch = un passage complet sur les données = plusieurs pas. 
Ex: 400 textes, batch 32  (400/32)→ 13 pas(eta) par epoch (le 13/13 de Keras)

Le batch_size ne concerne que la partie d'entrainement(train) qui a prealablement ete decoupe en train/test.


Les features d'entree et de sortie son decorrelées ; les entrées sont les label ( bigramme) et la sortie est le nombre de langues que je souhaite identifié.
