import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

NOMES_CLASSES = ['crítico', 'instável', 'pot. estável', 'estável']


def avaliar(modelo, X_teste, y_teste, nome_modelo='modelo'):
    y_pred = modelo.predict(X_teste)

    acuracia  = accuracy_score(y_teste, y_pred)
    precisao  = precision_score(y_teste, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_teste, y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_teste, y_pred, average='macro', zero_division=0)

    print(f'\n=== {nome_modelo} ===')
    print(f'Acurácia : {acuracia:.4f}')
    print(f'Precisão : {precisao:.4f}  (macro)')
    print(f'Recall   : {recall:.4f}  (macro)')
    print(f'F1-score : {f1:.4f}  (macro)')
    print()
    print(classification_report(y_teste, y_pred, target_names=NOMES_CLASSES, zero_division=0))

    plotar_matriz_confusao(y_teste, y_pred, nome_modelo)

    return {'acuracia': acuracia, 'precisao': precisao, 'recall': recall, 'f1': f1}


def plotar_matriz_confusao(y_teste, y_pred, nome_modelo):
    cm = confusion_matrix(y_teste, y_pred, labels=[1, 2, 3, 4])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=NOMES_CLASSES,
                yticklabels=NOMES_CLASSES)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de Confusão — {nome_modelo}')
    plt.tight_layout()

    nome_arquivo = nome_modelo.lower().replace(' ', '_')
    caminho = f'results/plots/cm_{nome_arquivo}.png'
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f'Matriz de confusão salva em: {caminho}')
