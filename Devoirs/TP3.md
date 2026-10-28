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

##Remarques :
En laissant le model tel quel (tp2_cours_2), appliqué a ce nouveau corpus unilingue qui contient des label qui categorisent le type d'article de press, il a tendance a faire de l'overfitting.
Tester les propositions d'amelioration suggeré en cours :
- modification de l'architechteure du model:
         - embedder les données avec keras et modifer le **output_mode="int"** au niveau de la **TextVectorization**.
         - adapter le script en creant un dictionnaire "Classes" dans lequel j'attribu un encodage pour chaque categories.
         - dans mon modele l'optimizer Adam ou SGD = de ce que j'ai compris adam est fondé sur SGd mais il est plus rapide??
  Mes resultats avec SGD sont nulles : le modele n'apprend rien...  il est constitu de la façon suivante
  
  '''
  
        def create_model(vocab_size, nb_classes, dim=64, optim="SGD", dropout=0.4):
    #j'ai choisi un dropt out moderer pour ne pas perdre trop d'information entre chaque apprentissage
        model = Sequential([
            keras.layers.Input(shape=(500,)),
            keras.layers.Embedding(vocab_size, dim, mask_zero=True) #j'embedde les données du corpus ,
            keras.layers.GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(nb_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=optim,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
    

  '''
  
  - j'ai  eu un  petit probleme de version avec tensorflow et keras : 
  sur stack overflow ils recommendent de modifier les version dans mon Environnement : 
  
  ''' 
  
        pip install tensorflow==2.15.0 tensorflow-hub keras==2.15.0

  '''
  
  - j'ai essayé GlobalAveragePooling1D = 0.14685314893722534 precision
                GlobalMaxPooling1D = 0.14685314893722534 toujours nulle
    Selon la matrice de confusion le modele a tendance a tout mettre dans societe ???
- Pour les premiers essaie meme si j'ai instancé mon drop out a 0.4 je ne l'utilise pas encore .


''' 

        def create_model(vocab_size, nb_classes, dim=64, optim="SGD", dropout=0.4):
    model = Sequential([
        keras.layers.Input(shape=(500,)),
        keras.layers.Embedding(vocab_size, dim, mask_zero=True),
        keras.layers.GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        keras.layers.Dropout(dropout),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        keras.layers.Dropout(dropout),
        Dense(8, activation='relu'),
        Dense(nb_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=optim,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

'''
Meme en ajoutant un dropout de 40%  le modele a une precision nulle , et il classe toujours tout dans la meme Categories???

Je re essaie en laissant un seul drop out (modifié a 30%) et en modifiant par le GlobalAveragePooling1D aussi moins de couches dense...

Mon accuracy est toujours nulle : 0.14685314893722534 ..

J'ai changé SGD par AdamW + j'ai augmente le nombre d'epoch à 300: et ma precision a prit +40 % (envirion) 0.5664335489273071
Mais .. le modele n'apprend toujours pas En revance, la bonne nouvellle : il ne classe plus tous dans Societe !!

J'ai mis 1 drop out et une seule couche dense et passer a 5000 la shape du input et mon accuracy etait de 0.5734265446662903.
C'est toujours mieux qu'au debut mais il ya quand meme un grand sur apprentissage  dans l'entrainement ....


**Et si** le probleme ne viendrait pas des données en elles memes ?? peut etre que ce n'est pas assez suffisant pour que le model apprenne suffisamment sans faire d'overfitting..
