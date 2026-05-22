import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import carregar_dados, dividir_dados, normalizar
from src.random_forest import treinar as treinar_rf
from src.mlp import treinar as treinar_mlp
from src.evaluation import avaliar

df = carregar_dados()
X_treino, X_teste, y_treino, y_teste = dividir_dados(df)
X_treino_norm, X_teste_norm = normalizar(X_treino, X_teste)

print('=== RANDOM FOREST (C4.5) ===')
modelo_rf = treinar_rf(X_treino, y_treino)
avaliar(modelo_rf, X_teste, y_teste, nome_modelo='Random Forest')

print('=== MLP ===')
modelo_mlp = treinar_mlp(X_treino_norm, y_treino)
avaliar(modelo_mlp, X_teste_norm, y_teste, nome_modelo='MLP')
