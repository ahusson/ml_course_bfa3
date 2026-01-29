# TD Arbres de Décision - M2 Banque Finance Assurance
## Paris Dauphine - Version Enrichie

## 📦 Contenu du Package

Ce package contient tous les éléments pour un TD complet de 3h sur les arbres de décision :

### 1. **TD_Decision_Trees_M2_Dauphine_BASE.ipynb** (Notebook de base)
Le notebook principal avec :
- ✅ Imports et configuration
- ✅ Partie 1 : Théorie (30 min) - version concise
- ✅ Partie 2 : Exercices guidés (1h)
  - Exercice 1 : Classification de défaut de crédit (15 min)
  - Exercice 2 : Détection de fraude bancaire (15 min)
  - Exercice 3 : Pruning et régularisation (15 min)
  - Exercice 4 : Interprétabilité (15 min)
- ✅ Partie 3 : Projet non guidé (1h30) - German Credit Data

### 2. **TD_Arbres_Supplement_Theorique.pdf** (Support théorique enrichi)
Document PDF de 6-8 pages contenant :
- ✅ **Section 1 : Critères de Division détaillés**
  - Indice de Gini : origine, formule, exemples avec calculs complets
  - Entropie de Shannon : théorie de l'information, exemples
  - Graphique comparatif Gini vs Entropie
  - Gain d'information : calcul étape par étape
  
- ✅ **Section 2 : Hyperparamètres et Interprétabilité**
  - Pre-pruning : exemples concrets avec impact sur l'interprétabilité
  - Post-pruning : exemples visuels avec différents alphas
  - Tableaux comparatifs
  
- ✅ **Section 3 : Frontières Orthogonales**
  - Explication avec visualisation
  - Limitations et solutions

### 3. **script_visualisation_apprentissage.py** (À intégrer)
Script Python complet pour visualiser le processus d'apprentissage :
- 🎯 Génération de données synthétiques 2D
- 📊 Fonction `visualize_iteration()` qui crée 3 graphiques :
  1. Données + frontière de décision
  2. Gini de chaque région
  3. Gains d'information des divisions
- 🔄 Visualisation pour 3 itérations (profondeurs 1, 2, 3)

**À ajouter dans le notebook après la section théorique 1.2**

### 4. **script_gridsearch_exercice.py** (À intégrer)
Script Python pour l'exercice GridSearch :
- 🔍 GridSearch sur pre-pruning (4 hyperparamètres)
- 🔍 GridSearch sur post-pruning (ccp_alpha)
- 📈 Visualisations de l'impact
- 📊 Fonction de comparaison des modèles

**À ajouter comme Exercice 3 dans la Partie 2 du notebook**

## 🎯 Comment utiliser ce package

### Option 1 : Utilisation rapide (notebook de base)
1. Ouvrez `TD_Decision_Trees_M2_Dauphine_BASE.ipynb`
2. Distribuez `TD_Arbres_Supplement_Theorique.pdf` comme support théorique
3. Les étudiants peuvent suivre le TD directement

### Option 2 : Version complète enrichie
1. Copiez le contenu de `script_visualisation_apprentissage.py` dans le notebook après la section 1.2
2. Copiez le contenu de `script_gridsearch_exercice.py` pour créer l'Exercice 3
3. Résultat : notebook complet avec visualisations interactives

### Option 3 : Utilisation modulaire
- Utilisez le PDF comme **support de cours projeté** pendant les 30 min de théorie
- Utilisez le notebook pour les **exercices pratiques**
- Les scripts peuvent servir de **correction détaillée**

## 📝 Modifications apportées par rapport au notebook original

### Ajouts théoriques (dans le PDF) :
1. ✅ **Explication détaillée des critères Gini et Entropie**
   - D'où viennent ces formules (probabilité de mal classer, théorie de l'information)
   - Que représentent les p_i (proportions des classes)
   - Exemples numériques complets avec 100 clients

2. ✅ **Visualisation du processus d'apprentissage**
   - 3 itérations illustrées
   - 3 graphiques par itération
   - Calculs des gains affichés

3. ✅ **Hyperparamètres et interprétabilité détaillés**
   - Exemples concrets pour chaque paramètre
   - Impact sur le nombre de règles
   - Cas d'usage (exploration vs production)

### Ajouts pratiques (scripts à intégrer) :
1. ✅ **Exercice GridSearch complet**
   - Pre-pruning : 4 hyperparamètres × plusieurs valeurs
   - Post-pruning : recherche optimale de ccp_alpha
   - Visualisations de l'impact
   - Comparaison finale des approches

2. ✅ **Visualisations interactives**
   - Code pour générer les graphiques d'apprentissage
   - Fonctions réutilisables
   - Commentaires pédagogiques

## 🎓 Pédagogie

### Structure maintenue (3h) :
- **30 min** : Théorie (avec PDF comme support)
- **1h** : 4 exercices guidés (dont GridSearch)
- **1h30** : Projet German Credit Data

### Points pédagogiques renforcés :
1. **Compréhension profonde** des critères (pas juste les formules)
2. **Visualisation** du processus itératif (pas de boîte noire)
3. **Optimisation systématique** avec GridSearch (pas de tuning manuel)
4. **Lien avec la finance** (réglementation, interprétabilité)

## 💻 Prérequis techniques

```python
# Packages requis
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter

# Installation
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 📊 Données

- **Exercices guidés** : Données synthétiques générées dans le notebook
- **Projet final** : German Credit Data (téléchargement automatique depuis UCI)

## 🔧 Personnalisation

### Pour adapter le TD :
1. **Durée** : Ajuster le nombre d'exercices (4 actuellement)
2. **Difficulté** : Modifier les TODO pour plus/moins de guidage
3. **Thématique** : Changer les exemples (crédit, fraude, assurance...)
4. **Dataset final** : Remplacer German Credit par vos propres données

### Fichiers modifiables :
- Notebook : `.ipynb` (format standard Jupyter)
- PDF : Généré depuis le script Python (modifiable)
- Scripts : `.py` (Python standard)

## 📚 Ressources complémentaires

### Dans le notebook :
- Liens vers documentation scikit-learn
- Références académiques (Breiman, Hastie)
- Suggestions pour aller plus loin (Random Forests, SHAP)

### Pour l'enseignant :
- Tous les TODO ont une solution intégrée
- Commentaires expliquent les choix pédagogiques
- Timing indicatif pour chaque section

## ✨ Points forts de cette version

1. **Théorie approfondie** sans alourdir le notebook
2. **Visualisations interactives** du processus d'apprentissage
3. **GridSearch intégré** (pratique industrielle)
4. **Interprétabilité** comme fil rouge (crucial en finance)
5. **Mix équilibré** théorie/pratique/projet

## 🐛 Support et Questions

Pour toute question sur l'utilisation de ce matériel pédagogique :
- Les scripts sont commentés ligne par ligne
- Le PDF contient les explications détaillées
- Le notebook inclut des "Questions de réflexion"

## 📄 Licence

Matériel pédagogique pour usage académique.
Paris Dauphine - Master 2 Banque Finance Assurance

---

**Version** : 2.0 Enrichie  
**Date** : Janvier 2026  
**Auteur** : Matériel généré pour Paris Dauphine M2

Bon TD ! 🚀
