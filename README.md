# 📌 Prédiction de l’Attrition Client dans le Secteur Bancaire

Projet présenté par Bilal SAYOUD

🎯 Objectif du projet

L’attrition client est un enjeu stratégique pour les banques.
Identifier en amont les clients susceptibles de quitter permet :

d’anticiper les pertes de revenus

d’améliorer la fidélisation

d’augmenter la satisfaction client

Le but du projet est de maximiser la ROC-AUC afin d’identifier de manière fiable les clients à risque.

🗂️ Contenu du projet
1. Préparation des données

Analyse exploratoire (EDA)

Nettoyage & détection des outliers

Transformation des données

Feature engineering avancé

Normalisation & encodage

SMOTE pour équilibrer la cible

2. Optimisation des modèles

Modèles testés :

RandomForest, ExtraTrees

LightGBM, XGBoost, CatBoost

MLPClassifier

Logistic Regression

Techniques :

Validation croisée stratifiée

Filtrage des folds avec test de Kolmogorov-Smirnov

Hyperparameter Tuning : GridSearch, Optuna

Sélection de variables par permutation importance et corrélations

SMOTE

Normalisation

One-hot encoding


