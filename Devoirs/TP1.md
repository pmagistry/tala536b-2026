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
## Notes du tp:

J'ai tester en ajoutant un bi gramme pour l'espagnol et modifier les dimensions des cours du reseau et la precision est plutot correct ? ( est ce que c'est bizarre ?)
Compilation du modèle | j'ai modifier la (dense) de la sortie n'est plus simplement 0 ou 1 car on est plus sur une classification binaire.

Donc si j'ai bien compris le reseau de neuronne aurait pu aussi garder 3 fearture et 2 langue a identifier ( donc une sortie binaire ) et faire le comptage normalisé  et l'apprentissage en ajustant les poids et en fonction de ce qu'il aura identifie il attribura le label allemand ou anglais par exemple ( label encodé 0 ou 1).
