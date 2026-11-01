# Classification de texte par catégories (ou journal source) avec couche TextVectorization

## Expliquer ci-dessous vos difficultés ou posez vos questions

- J'ai mis beaucoup de temps à trouver le bon moyen d'envoyer les vecteurs au layer Input avec la bonne taille : finalement, j'ai choisi de donner une longueur de séquence fixe à mes textes vectorisés avec le paramètre _output_sequence_length_ passé à TextVectorization fixé à 300. Les phrases ayant plus de 300 tokens seront tronquées et celles de moins de 300 tokens ont un padding (0 à la place des tokens manquants) (https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization?version=stable#args_1).
- Je ne suis pas sûr de l'endroit où la couche GlobalAveragePooling1D est la mieux placée
- Premier test : presque tous les articles ont la prédiction 'science'. Je n'arrive pas à savoir si c'est un hasard ou l'architecture du modèle qui donne la réponse 3 (mappée à science) parce que c'est pile entre 0 et 6
- j'ai eu des difficultés à bien savoir comment découper le corpus en train, test et validation. J'ai choisi la méthode de découper deux fois avec sklearn
- j'ai trouvé le bug : la couche normalization après l'input (et donc après textvectorization) n'a pas de sens après la vectorization des tokens. Elle modifie énormément les données et les rend complètement fausses car les vecteurs sont des plongements lexicaux qui ne doivent pas être normalisés pour toujours représenter les bons tokens. Après avoir enlevé la couche, l'accuracy monte à 93% pendant le training puis 54% sur le test
- augmenter la valeur de la couche Dropout de 0.7 à 0.8 donne une accuracy au training plus basse (0.8) mais une validation accuracy plus haute qui correspond mieux à la training accuracy. On évite alors le sur-apprentissage
- je me demande si ça peut arriver que la validation accuracy et la test accuracy soient complètement différentes et ce que ça voudrait dire si ça arrivait
- j'ai également choisi adam comme optimizer pour avoir un learning_rate qui s'adapte. Sans Adam et juste Stochastic Gradient Descent et un learning_rate par défaut (0.1 je crois), on a une accuracy proche du hasard.
