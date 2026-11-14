# Expériences

Scripts lancés pour choisir le modèle du notebook `notebooks/v2_meilleur_modele.ipynb`,
gardés tels qu'ils ont tourné (ils ne sont pas vérifiés par ruff). Chaque script lit
le train et la validation via `exp.py` ; le test n'est jamais lu ici.

À lancer depuis la racine du dépôt, après `make_splits.py` et `pip install -e ".[experiments]"` :

```bash
python experiments/r1.py            # puis r1b.py, r2.py extra, r2.py tfidf, ...
python experiments/r3_tune.py lgbm all+extra+tfidf 20   # modèle, variables, minutes
```

| Script | Piste |
|---|---|
| `exp.py` | banc d'essai : validation croisée à 5 plis sur le train, score en validation, journal `out/results.csv` |
| `urlfe.py` | premières versions des variables d'URL, du TF-IDF et de l'empilement (devenues `phishing.url_text`) |
| `r1.py`, `r1b.py` | 14 modèles de référence, sous-ensembles de variables, premiers plis par domaine |
| `r2.py` | 70 variables lexicales ajoutées, TF-IDF de caractères et de mots |
| `r2b.py` | probabilité TF-IDF empilée hors pli |
| `r2c.py` | fuite de domaine : plis internes et externes groupés par domaine enregistré |
| `r4_text.py`, `r5.py` | variantes du modèle texte (mots, hôte/chemin, SVM linéaire) |
| `r3_tune.py` | réglage Optuna (log loss en validation croisée), résultats dans `out/best_*.json` |
| `models.py`, `r6.py` | boostings réglés et vote souple |
| `r7.py` | modèle URL seule, ExtraTrees, empilement, encodage cible du domaine |
| `r8.py` | validation croisée répétée 3 × 5 des finalistes |

## Journal

Toutes les lignes sont mesurées de la même façon : validation croisée à 5 plis sur le
train seul (3 répétitions pour les trois dernières lignes), puis modèle entraîné sur le train
et évalué sur la validation. « Par domaine » : plis sans domaine commun. « 81 var. du jeu » :
les 87 variables du jeu moins les 6 constantes. « Ajoutées » : les 70 variables de
`phishing.url_text`. Les hyperparamètres « optimisés » viennent d'Optuna (log loss en
validation croisée sur le train).

| Modèle | Variables | Exactitude CV | AUC CV | Brier CV | Exactitude CV par domaine | Exactitude validation |
|---|---|---|---|---|---|---|
| Classe majoritaire | 81 var. du jeu | 50,01 % | 0,4999 | 0,2500 |  | 49,97 % |
| Naive Bayes gaussien | 81 var. du jeu | 64,85 % | 0,6585 | 0,3508 |  | 65,25 % |
| Arbre de décision | 81 var. du jeu | 92,96 % | 0,9611 | 0,0565 |  | 94,75 % |
| k plus proches voisins (k=15) | 81 var. du jeu | 93,32 % | 0,9805 | 0,0502 |  | 94,46 % |
| Régression logistique | 81 var. du jeu | 94,31 % | 0,9852 | 0,0426 |  | 95,34 % |
| Perceptron multicouche (128, 64) | 81 var. du jeu | 95,54 % | 0,9887 | 0,0346 |  | 96,21 % |
| SVM à noyau RBF | 81 var. du jeu | 95,80 % | 0,9894 | 0,0323 |  | 96,21 % |
| ExtraTrees | 81 var. du jeu | 96,37 % | 0,9935 | 0,0324 |  | 96,97 % |
| Forêt aléatoire | 81 var. du jeu | 96,60 % | 0,9934 | 0,0325 |  | 96,73 % |
| CatBoost | 81 var. du jeu | 96,64 % | 0,9935 | 0,0260 |  | 97,32 % |
| HistGradientBoosting | 81 var. du jeu | 96,72 % | 0,9934 | 0,0265 |  | 97,38 % |
| XGBoost | 81 var. du jeu | 96,84 % | 0,9938 | 0,0254 |  | 97,14 % |
| LightGBM | 81 var. du jeu | 97,12 % | 0,9943 | 0,0245 | 95,12 % | 97,38 % |
| LightGBM | 26 var. recalculables depuis l'URL | 86,79 % | 0,9447 | 0,0938 |  | 85,83 % |
| LightGBM | variables URL | 91,36 % | 0,9711 | 0,0642 |  | 91,72 % |
| LightGBM | URL + contenu | 94,29 % | 0,9850 | 0,0432 |  | 94,58 % |
| LightGBM | contenu + externes | 95,00 % | 0,9871 | 0,0393 |  | 95,45 % |
| LightGBM | URL + externes | 96,94 % | 0,9932 | 0,0258 |  | 97,08 % |
| TF-IDF mots + rég. log. | URL brute | 92,24 % | 0,9706 | 0,0614 |  | 93,12 % |
| TF-IDF caractères 2-6 + rég. log. | URL brute | 92,85 % | 0,9804 | 0,0533 |  | 93,88 % |
| TF-IDF caractères 1-5 + rég. log. | URL brute | 93,00 % | 0,9813 | 0,0518 |  | 94,29 % |
| TF-IDF caractères + mots | URL brute | 94,05 % | 0,9830 | 0,0459 | 91,01 % | 94,58 % |
| SVM linéaire calibré, TF-IDF | URL brute | 94,14 % | 0,9825 | 0,0458 | 91,04 % | 94,40 % |
| TF-IDF hôte / chemin / mots | URL brute | 94,36 % | 0,9849 | 0,0429 | 91,04 % | 94,93 % |
| LightGBM | recalculables + 70 var. ajoutées | 92,09 % | 0,9752 | 0,0586 |  | 92,48 % |
| LightGBM optimisé | recalculables + ajoutées + TF-IDF (URL seule) | 94,62 % | 0,9873 | 0,0399 | 92,39 % | 95,51 % |
| LightGBM | 81 + 70 var. ajoutées | 97,16 % | 0,9945 | 0,0228 | 95,61 % | 97,38 % |
| CatBoost | 81 + 70 var. ajoutées | 96,84 % | 0,9938 | 0,0248 |  | 97,49 % |
| LightGBM | 81 + TF-IDF empilé | 97,47 % | 0,9963 | 0,0203 |  | 97,84 % |
| LightGBM | 81 + ajoutées + TF-IDF (plis internes aléatoires) | 97,61 % | 0,9963 | 0,0198 | 96,46 % | 98,02 % |
| LightGBM | 81 + ajoutées + TF-IDF (plis internes par domaine) | 97,71 % | 0,9963 | 0,0194 | 96,45 % | 97,96 % |
| LightGBM | 81 + ajoutées + TF-IDF hôte/chemin | 97,69 % | 0,9960 | 0,0197 | 96,12 % | 98,02 % |
| LightGBM optimisé | + encodage cible du domaine | 97,65 % | 0,9963 | 0,0185 | 96,66 % | 97,73 % |
| CatBoost optimisé | 81 + ajoutées + TF-IDF | 97,52 % | 0,9961 | 0,0196 |  | 97,67 % |
| XGBoost optimisé | 81 + ajoutées + TF-IDF | 97,70 % | 0,9962 | 0,0185 |  | 97,84 % |
| LightGBM optimisé | 81 + ajoutées + TF-IDF | 97,67 % | 0,9962 | 0,0187 |  | 97,90 % |
| Empilement (rég. log. sur 4 modèles) | 81 + ajoutées + TF-IDF | 97,81 % | 0,9958 | 0,0185 |  | 97,78 % |
| Vote LightGBM + XGBoost + CatBoost + ExtraTrees | 81 + ajoutées + TF-IDF | 97,73 % | 0,9963 | 0,0183 | 96,76 % | 97,61 % |
| **Vote LightGBM + XGBoost + CatBoost (retenu)** | 81 + ajoutées + TF-IDF | 97,73 % | 0,9963 | 0,0183 | 96,76 % | 97,73 % |

Ce qui ressort :

- les boostings dominent dès le départ ; la régression logistique est près de 3 points derrière ;
- ajouter les 7 variables externes aux variables d'URL fait passer LightGBM de 91,4 % à 96,9 % ;
- l'ajout qui compte le plus est la probabilité TF-IDF empilée (+0,5 point en plis
  aléatoires, +0,8 point en plis par domaine, par rapport aux 81 + 70 variables) ;
- l'optimisation des hyperparamètres et les ensembles gagnent surtout en Brier ;
- l'encodage cible du domaine n'apporte rien de plus que le TF-IDF : on ne le garde pas ;
- au-delà de 97,7 %, les écarts sont dans le bruit (environ ±0,2 point entre répétitions).