import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUNAS_TREINO_COM_LABEL = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']

FEATURES = ['qPA', 'pulso', 'resp']
TARGET = 'classe'


def carregar_dados(caminho='data/02_treino_sinais_vitais_com_label.txt'):
    df = pd.read_csv(caminho, header=None, names=COLUNAS_TREINO_COM_LABEL)
    df = df.set_index('id')
    df = df.drop(columns=['pSist', 'pDiast', 'gravidade'])
    return df


def dividir_dados(df):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_treino, X_teste, y_treino, y_teste


def normalizar(X_treino, X_teste):
    scaler = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_teste_norm  = scaler.transform(X_teste)
    return X_treino_norm, X_teste_norm
