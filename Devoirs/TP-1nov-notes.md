Sujet à rendre pour le 1er novembre:
## Consigne du forum.
Il faut maintenant essayer une stratégie différente qui commence par une couche d'Embedding(). Il faudra pour cela adopter un output_mode="int" au niveau de la TextVectorization. Cela permet d'obtenir des séquences de vecteurs (plongements) où chaque vecteur correspond à un token du texte.

Pour revenir à une tâche de classification (passer de la séquence à un vecteur unique dont la longueur dépend du nombre de catégories), il faudra essayer les couches GlobalMaxPooling1D ou GlobalAveragePooling1D


## Expliquer ci-dessous vos difficultés ou posez vos questions

###Les difficultés et problème pricipales
- N'ayant pas était présent le 20 je ne suis pas sur d'avoir bien compris quoi faire et quoi adapter . J'ai essayer de le faire en suivant les consignes, ce qui a était fait en cours et le idée donner par une ia pour avoir une direction.

### Notes sur ce qui à était fait
- Moddification du notebooke pour le donnée json.
- Teste de Moddification des finction pour comparrer avec et sans drop out et avec des rate différents
    -> Les resultats semble meilleur avec les drop out et encore plus avec lr plsu bas 

- Tentative et testes avec les embbedings 
    -> J'ai tester différent lr aussi mais n'ai aps vu de différence notable. J'ai essayer aussi de comparer entre un modèle avec pooling mx et avrages 

J'ai un tableau csvavec différents résultats de teste mais j'avous ne pas avoir tout bien compris quels choix serait le plus int"éressant a faire et quel modèles utiliser dans quel situation

|  | Loss test | Acc test |
|---|---|---|
| Base version Tp2 sur json | 1.812 | 0.301 |
| Mots 5000 | 2.606 | 0.371 |
| Mots 1000 | 3.710 | 0.329 |
| Mots 100 | 2.562 | 0.399 |
| Avec validation sans Dropout (test1) | 2.354 | 0.357 |
| Avec validation sans Dropout (test2) | 2.182 | 0.364 |
| Dropout 0.5 | 1.934 | 0.224 |
| Dropout 0.2 | 2.466 | 0.413 |
| Embedding avrg lr 0.1 | 1.910 | 0.168 |
| Embedding avrg lr 1.0 | 2.282 | 0.210 |
| Embedding max lr 1.0 | 1.838 | 0.462 |
