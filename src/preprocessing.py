import pandas as pd

COLUNAS_TREINO_COM_LABEL = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
COLUNAS_TREINO_SEM_LABEL = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade']

FEATURES = ['qPA', 'pulso', 'resp']
TARGET_REGRESSAO = 'gravidade'
TARGET_CLASSIFICACAO = 'classe'


def carregar_com_label(caminho='data/02_treino_sinais_vitais_com_label.txt'):
    df = pd.read_csv(caminho, header=None, names=COLUNAS_TREINO_COM_LABEL)
    df = df.set_index('id')
    df = df.drop(columns=['pSist', 'pDiast'])
    return df


def carregar_sem_label(caminho='data/01_treino_sinais_vitais_sem_label.txt'):
    df = pd.read_csv(caminho, header=None, names=COLUNAS_TREINO_SEM_LABEL)
    df = df.set_index('id')
    df = df.drop(columns=['pSist', 'pDiast'])
    return df
