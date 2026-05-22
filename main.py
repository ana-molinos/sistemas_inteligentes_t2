from src.preprocessing import carregar_dados, dividir_dados
from src.random_forest import treinar
from src.evaluation import avaliar

df = carregar_dados()
X_treino, X_teste, y_treino, y_teste = dividir_dados(df)

print('=== RANDOM FOREST (C4.5) ===')
modelo_rf = treinar(X_treino, y_treino)
avaliar(modelo_rf, X_teste, y_teste, nome_modelo='Random Forest')
