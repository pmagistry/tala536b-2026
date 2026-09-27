# Exercices du 22 septembre (à pusher le dimanche 27)

## Environnement de travail.

- [x] cloner le github ✅ 2026-09-27
- [x] se créer un environnement virtuel avec `uv`, installer les bibliothèques torch, keras, marimo, matplotlib, scikit-learn. ✅ 2026-09-27
- [x] essayer de lancer et de faire tourner le notebook de la première séance (`marimo edit cours1.py`) ✅ 2026-09-27

## Lecture et modification du code

- [x] s'assurer que l'on comprend le code du notebook ✅ 2026-09-27
- [x] essayer de reprendre le code en pytorch pour effectuer la classification de langues donnée en keras. ✅ 2026-09-27
	- ça veut dire évaluer la classification après entraînement ???
- [x] essayer d'ajouter une couche au réseau défini en Keras ✅ 2026-09-27
- [ ] essayer d'ajouter un troisième bigramme de lettre en entrée du réseau.
	- par exemple "de" (ex der, den, dem)

## Proposer des pistes (sans coder) pour classer des documents entre 3 langues ou plus.

...

## Expliquer ci-dessous vos difficultés ou posez vos questions

(ou dites si tout va bien !)
...

- que fait `.item()` ? (dans `train_perceptron`)
- pourquoi `y_col = y.unsqueeze(1)` ? (dans la définition de `build_and_train_single_neuron(X, y)`)
- on parle d'**ajouter une couche cachée non-linéaire**, mais pourtant dans le `__init__` de la classe MLP, on a : 
```python
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2) # cf W1 ?
        self.out = nn.Linear(2, 1) # cf W2
```
où l'attribut "hidden" est donc finalement une couche linéaire. C'est seulement dans la fonction `forward` qu'on trouve la tangente hyperbolique dont on parlait pour ajouter de la non-linéarité : 
```python
    def forward(self, x):
        return self.out(torch.tanh(self.hidden(x)))
```
Je ne comprends pas trop la structure, et pourquoi la section parle d'ajouter une couche cachée non-linéaire... je ne suis pas sûre non-plus d'à quoi correspond la fonction `forward`. J'ai l'impression que c'est celle qui sort la prédiction du modèle pour une entrée donnée...?

- pourquoi le calcul de la accuracy est si complexe à la fin ?
```python
# pseudo-evaluation
with torch.no_grad():
    acc_mlp = ((torch.sigmoid(mlp(X_xor)) > 0.5).float() == y_xor).float().mean().item()
```
La sigmoïde renvoie 0 ou 1 (pour le XOR, au résultat binaire). On regarde si c'est supérieur à 0.5 ou non, et on convertir ça en float.... donc en gros si on avait le modèle qui renvoie 0 ça donne finalement 0.0, et s'il prédit 1 on a finalement 1.0 (sous forme de tenseur PyTorch ?)... et on vérifie si c'est égal à y_xor (` == y_xor`) → on aura un tenseur avec des True là où on a bien deviné, False sinon. On le convertit en floats (`.float()`) donc on a un tenseurs de 1.0 (bien prédit) et 0.0 (mal prédit), dont on fait la moyenne (`.mean()`).  OK en décortiquant ça fait sens...
Après je ne sais pas à quoi sert `.item()`.
Et là cette fois on a : 
```python
    hidden_xor = torch.tanh(mlp.hidden(X_xor))
```
Donc ici oui `hidden_xor` a de la non-linéarité