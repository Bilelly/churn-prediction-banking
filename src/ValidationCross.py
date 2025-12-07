

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

# Définir un K-Fold stratifié 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fonction utilitaire pour afficher les résultats de la validation croisée
def cross_val_results(model, X, y):
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    print("----- Cross Validation Scores -----")
    for metric in scoring.keys():
        print(f"{metric}: {results[f'test_{metric}'].mean():.4f} ± {results[f'test_{metric}'].std():.4f}")
    print("\n")  # pour espacer les résultats