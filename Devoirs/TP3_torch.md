Sujet du 20 octobre, à pusher le lundi 2 novembre avant 12h.

## Environnement de travail.

- n'hésitez pas à avoir un fichier `.py` supplémentaire avec vos fonctions utilitaires pour décharger le notebook

## Classification en 6 Categories 

- Modifier le chargement des données pour bien avoir 6 categories fait 
- Modifier la structure réseau en conséquence (input et couche de sortie surtout, vous pouvez aussi faire varier la ou les couches cachées) fait 
- Modifier la fonction de loss en conséquence (et la dernière fonction d'activation en `softmax`) fait 
- Embedding fait 
-  ajouter/ tester GlobalMaxPooling1D ou GlobalAveragePooling1D fait 


## Consignes :

Il faut maintenant essayer une stratégie différente qui commence par une couche **d'Embedding()**. Il faudra pour cela adopter un **output_mode="int"** au niveau de la **TextVectorization**. Cela permet d'obtenir des séquences de vecteurs **(plongements)** où chaque vecteur correspond à un token du texte.

**Pour revenir à une tâche de classification (passer de la séquence à un vecteur unique dont la longueur dépend du nombre de catégories), il faudra essayer les couches GlobalMaxPooling1D ou GlobalAveragePooling1D**

## Remarques :

La modification de l'architechture en pytorch est certes plus complexe mais cela permet d'avoir un conrole total sur le model tandis que keras est un peu moins transparent sur certains aspect mais son avantage est sa "simplicité". Si j'ai bien compris on peu dire que l'architecture keras est "haut niveau" et celle de pytorch "haut niveau"?
car il y a des modules qui existent en keras et pas en pytorch comme pour le pooling il faut creer soit meme la fonction.

Pour reprendre chauque etapes des script utils keras et torch:

- Le chargement des données Json est identique pour les deux. **load_json()**

- La vectorisation du vocabulaire ; pour keras il y a simplement l'appel a l'api **TextVectorization**,
la tonkenisation est encapsulé. Tandis qu'en pytorch il faut creer une fonction vocabulaire et initialiser une matrice de vecteur a 0,
afin de pouvoir faire un padding? keras = **keras.layers.TextVectorization**  | torch = **construire_vocab()**+ **vectoriser()**

- Embedder les donées : en keras on a la fonction **create_model()** et en torch = **creer_model_()**.
Attention en keras on doit specifier l'activation tandis que pour pytorch elle fait partie **nn.CrossEntropyLoss()**.
+la version pytorch separe le classifieur **(tete)** de l'extracteur de caracteristique **(embedding)** alors que keras a tout ceci dans sequential. 

- pour la partie du forward +padding keras s'en charge implicitement tandis qu'avec torch on doit creer la fonction predir_embedding() qui generer un tenseur booleen (masque) et faire un produit matricielle de ceux là par les representations vectroriel le  .clamp(min=1) permet de ne pas faire de division par 0.

- L'entrainement est gerer par keras == history= model**.fit**(X_train, Y_train, epochs=300, batch_size=100, validation_split=0.1), ici fit s'occupe de tout.
pour la version torch il faut passer par la fonction  **entrainer_embedding()** qui a pour objectif de mettre à ajour les poids , en faisant une mise a zero des gradient pour eviter une explosion des poids.


