from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold


def treinar(X_treino, y_treino):
    modelo_base = RandomForestClassifier(
        criterion='entropy',
        random_state=10
    )

    param_grid = {
        'ccp_alpha':    [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        'n_estimators': [10, 50, 100, 150, 200]
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_treino, y_treino)

    print(f'Melhores parâmetros: {grid_search.best_params_}')
    print(f'Acurácia média na validação cruzada: {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_
