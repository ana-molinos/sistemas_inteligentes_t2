from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold


def treinar(X_treino, y_treino):
    modelo_base = MLPClassifier(
        max_iter=2000,
        random_state=42
    )

    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 10)],
        'activation':         ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01, 0.05]
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
