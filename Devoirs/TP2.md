Sujet du 6 octobre, à pusher le lundi 12 octobre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 langues

- [x] Modifier le chargement des données pour bien avoir 6 langues
- [x] Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées)
- [x] Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`)

- [ ] bonus (long): suivre le même principe en utilisant TextVectorizer de keras (on verra ça la semaine prochaine)

## Réaliser une évaluation propre

- [x] diviser correctement le corpus pour avoir des scores valides
- [ ] bonus: faire une matrice de confusion

## Expliquer ci-dessous vos difficultés ou posez vos questions

Le script principal pour ce TP se trouve dans ../Notebooks/tp2_classify_lang.py

La plus grosse difficulté pour moi est de décider comment envoyer le texte en input. Quelles sont les caractéristiques qui vont être pertinentes à extraire ?
Peut-être qu'avec le compte des trois bigrammes que j'avais déjà (th, en, wh), ce sera déja bien.
Je vais d'abord essayer de rajouter des bigrammes que je trouve discriminants.
Si je trouve d'autres caractéristiques pertinentes, je les rajouterai, sinon j'essaierai TextVectorizer.

En rajoutant des bigrammes, on fait passer l'accuracy du modele de environ 50 à 75%. J'aimerais bien pouvoir rapidement savoir pour quelles langues le modèle est le plus efficace, pour mieux savoir quels bigrammes essayer en plus.

En ajoutant une couche cachee, on fait monter l'accuracy a environ 88% mais elle baisse si on ajoute plus de couches cachees. Sauf si on ajoute une couche avec une fonction d'activatiom (comme relu).
J'ai essayé l'optimizer Adam mais ça n'améliore pas spécialement l'accuracy

- output : distribution de probabilités entre 6 classes (6 langues différentes) : ["eng", "deu", "nld", "fra", "ita", "spa"]
- donc fonction de loss choisie : Categorical Cross-Entropy, et fonction d'activation softmax comme conseillé pour avoir une distribution de probabilités

probleme avec pytorch : j'ai d'abord créé mon nn avec en derniere couche nn.Softmax(). J'ai compris plus tard que ce n'etait pas compatible avec la fonction de loss nn.CrossEntropyLoss() qui execute déjà la fonction softmax sur les sorties de la derniere couche.

Après avoir divisé le corpus en train et test, on réalise que l'accuracy descend d'environ 10% mais on a sûrement baissé le risque d'overfitting
