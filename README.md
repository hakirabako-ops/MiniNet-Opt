# MiniNet-Opt

Projet intégrateur du cours **Optimisation pour le Machine Learning** (
Auteure : BAKO Ange Stève Hakira.)

Le projet construit un petit réseau de neurones en NumPy, sans différentiation automatique, puis l'utilise pour étudier trois questions : comment l'optimiser (SGD, Momentum, Adam), comment contrôler sa généralisation (pénalité L2, dropout, early stopping), et ce que change une contrainte explicite sur la norme des poids, résolue par gradient projeté. PyTorch sert seulement à la fin, pour vérifier que les résultats NumPy sont corrects.

Le problème étudié est une classification binaire sur `make_moons` (1 500 points, bruit 0.20), avec un réseau 2 → 16 → 8 → 1 (ReLU, ReLU, sigmoïde) et la binary cross-entropy.


## Organisation du dépôt

```
Mon projet/
├── src/                     code du réseau, écrit en NumPy
│   ├── data.py              génération des données, split 60/20/20, standardisation
│   ├── layers.py            couches Linear, ReLU, Sigmoid, Dropout
│   ├── model.py             assemblage du réseau (MLP), forward et backward
│   ├── losses.py            binary cross-entropy
│   ├── optimizers.py        SGD, Momentum, Adam
│   ├── regularization.py    pénalité L2 et early stopping
│   ├── constraints.py       norme, projection sur la boule, gradient projeté
│   ├── train.py             boucle d'entraînement
│   └── plots.py             fonctions de tracé
├── notebooks/               une partie du sujet par notebook
│   ├── 01_data.ipynb                     partie A : données
│   ├── 02_layers_forward.ipynb           partie B : propagation avant
│   ├── 03_backward_gradcheck.ipynb       partie B : rétropropagation et vérification des gradients (J1)
│   ├── 04_optimizers.ipynb               partie C : SGD, Momentum, Adam
│   ├── 05_comparison_optimizers.ipynb    partie C : comparaison des optimiseurs (J2)
│   ├── 06_regularization.ipynb           partie D : régularisation, sélection et test (J3)
│   ├── 07_constraints.ipynb              partie E : contrainte et gradient projeté
│   ├── 08_penalty_vs_constraint.ipynb    partie F : pénalité ou contrainte (J4)
│   ├── 09_pytorch_check.ipynb            partie G : vérification avec PyTorch
│   └── 10_final_synthesis.ipynb          rapport final (J5)
├── results/                 tableaux (.csv), historiques (.pkl) et figures produits par les notebooks
└── README.md
```


## Installation

Il faut **Python 3.10** ou plus récent.

1. Ouvrir un terminal dans le dossier du projet.

2. Créer un environnement virtuel et l'activer :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # sous Windows : .venv\Scripts\activate
   ```

3. Installer les bibliothèques :

   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

4. Installer PyTorch, utilisé seulement par le notebook 09. La version CPU suffit et elle est beaucoup plus légère que la version standard (environ 200 Mo contre 3 Go) :

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

   Sans PyTorch, tous les notebooks fonctionnent sauf le 09.

5. Vérifier que tout est en place :

   ```bash
   python -c "import numpy, sklearn, matplotlib, pandas, torch; print('ok')"
   ```


## Exécution

Lancer Jupyter depuis le dossier du projet :

```bash
jupyter notebook
```

Ouvrir ensuite les notebooks du dossier `notebooks/` **dans l'ordre, de 01 à 10**, et exécuter chacun en entier (menu *Run All*). L'ordre compte : les notebooks enregistrent leurs résultats dans `results/`, et certains réutilisent ceux des précédents. Le rapport (notebook 10) relit tous ces fichiers.

Les notebooks doivent être lancés depuis le dossier `notebooks/` : ils importent le code avec `sys.path.append('..')` et écrivent dans `../results/`.

**Durée.** Les notebooks 01 à 04 et 10 s'exécutent en quelques secondes. Les notebooks 05 à 09 entraînent de nombreux modèles (5 seeds, jusqu'à 500 époques chacun) et prennent chacun plusieurs minutes. Une cellule longue n'est pas bloquée : il suffit d'attendre qu'elle se termine.

On peut aussi exécuter un notebook sans l'ouvrir, par exemple :

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 06_regularization.ipynb
```


## Reproductibilité

Les données sont générées avec la graine 42 et le découpage reste le même dans toutes les expériences. Chaque expérience est répétée sur les seeds 0 à 4, qui fixent l'initialisation des poids et l'ordre des mini-batchs. Relancer un notebook redonne donc exactement les mêmes chiffres.

Les hyperparamètres sont choisis sur le jeu de validation uniquement. Le jeu de test n'est utilisé qu'une fois, à la fin des notebooks 06 et 08. Les cellules qui évaluent le test refusent de s'exécuter deux fois dans la même session. Pour les relancer, il faut redémarrer le noyau (*Kernel > Restart*) puis exécuter tout le notebook depuis le début.


## Rapport

Le rapport complet se trouve dans `notebooks/10_final_synthesis.ipynb`. Il contient la dérivation du forward et du backward, la description des méthodes, le protocole expérimental, les résultats de chaque partie et les réponses aux douze questions du sujet. Pour obtenir une version PDF, on peut l'exporter depuis Jupyter (*File > Save and Export Notebook As > PDF*) ou en ligne de commande :

```bash
cd notebooks
jupyter nbconvert --to pdf 10_final_synthesis.ipynb
```

L'export PDF demande une installation de LaTeX. À défaut, l'export HTML (`--to html`) fonctionne sans rien installer de plus, et la page peut ensuite être imprimée en PDF depuis le navigateur.