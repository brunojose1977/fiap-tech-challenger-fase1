'''
Isso é um ótimo projeto! Como especialista em Machine Learning, Python e Ciência de Dados, vou guiar você através das etapas para carregar o conjunto de dados "diabetes.csv", realizar o pré-processamento necessário, e aplicar um modelo de Regressão Logística.

Para este projeto, usaremos as bibliotecas padrão de ciência de dados: pandas para manipulação de dados, numpy para operações numéricas, e scikit-learn (sklearn) para pré-processamento e modelagem.

1. Carregamento e Análise Inicial do Dataset
O primeiro passo é carregar o arquivo e inspecioná-lo. Assumirei que o arquivo diabetes.csv está no mesmo diretório que o seu script Python ou notebook.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Carregar o dataset
# try:
#     df = pd.read_csv('/home/brunojose/devops/python/TechChallenger1/datasetdiabetes.csv')
#     print("Dataset carregado com sucesso!")
# except FileNotFoundError:
#     print("Erro: O arquivo 'diabetes.csv' não foi encontrado. Certifique-se de que ele está no diretório correto.")
#     # Usar um dataset de exemplo para demonstração se necessário, mas o código abaixo assume o carregamento.
#     # return 
df = pd.read_csv('/home/brunojose/devops/python/TechChallenger1/dataset/diabetes.csv')
print("Dataset carregado com sucesso!")

# Visualizar as primeiras linhas e informações gerais
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações gerais (Tipos de dados e contagem de não-nulos):")
print(df.info())

print("\Estatísticas descritivas (incluindo Mínimo e Máximo):")
print(df.describe())

'''
2. Tratamento de Dados (Limpeza e Imputação)
No conjunto de dados Pima Indians Diabetes (o que é comumente usado com esta estrutura), algumas colunas como Glucose, BloodPressure, SkinThickness, Insulin, e BMI podem ter valores zero, o que biologicamente é impossível ou altamente improvável (exceto talvez em condições extremas ou erro de medição). Estes zeros são frequentemente tratados como valores nulos (NaN).

Identificação de Zeros Improváveis
Valores de 0 são suspeitos e serão substituídos por NaN para que possamos tratá-los corretamente.
'''
# Colunas a serem verificadas para valores 0 que representam nulos
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Substituir 0 por NaN nas colunas selecionadas
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

print("\nContagem de valores nulos (NaN) após substituição de zeros:")
print(df.isnull().sum())

'''
ratamento de Valores Nulos (Imputação)
O método mais comum para imputação é usar a mediana (menos sensível a outliers do que a média) dos dados existentes para preencher os valores ausentes (NaN).
'''

# Imputação de valores nulos
for col in cols_to_replace:
    # Preencher os NaNs com a mediana da respectiva coluna
    df[col].fillna(df[col].median(), inplace=True)

print("\nContagem de valores nulos (NaN) após imputação:")
print(df.isnull().sum())

'''
3. Preparação para o Modelo (Separação e Escalamento)
3.1. Definição de Features (X) e Target (y)
Separamos as variáveis preditoras (features) do alvo (target).
'''
# Separar Features (X) e Target (y)
X = df.drop('Outcome', axis=1) # Todas as colunas exceto 'Outcome'
y = df['Outcome']              # A coluna 'Outcome'

print(f"\nDimensão de X (Features): {X.shape}")
print(f"Dimensão de y (Target): {y.shape}")

'''
3.2. Divisão em Treino e Teste
É crucial dividir o dataset para treinar o modelo em um subconjunto e testar seu desempenho em dados nunca vistos.
'''

# Dividir o dataset em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y) # stratify garante que a proporção de 0s e 1s seja mantida

print(f"Dimensão de X_train: {X_train.shape}")
print(f"Dimensão de X_test: {X_test.shape}")

'''
3.3. Padronização e Normalização
O Escalamento de Dados é vital para modelos baseados em distância ou otimização, como a Regressão Logística, pois evita que features com grandes ranges dominem a função de custo.

Regras:

Padronização (StandardScaler): Transforma os dados para ter média 0 e desvio padrão 1. É ideal quando a distribuição dos dados se assemelha a uma normal. É a mais comum para Regressão Logística.

Normalização (MinMaxScaler): Transforma os dados para um range fixo (tipicamente 0 a 1). É útil quando você precisa de valores estritamente positivos (por exemplo, para Redes Neurais).

Sugerida: Para Regressão Logística, a Padronização (StandardScaler) é frequentemente a melhor escolha.
'''
# 1. Escolher o escalador (StandardScaler é a sugestão)
scaler = StandardScaler() 
# Ou usar MinMaxScaler() para normalização

# 2. Treinar o scaler APENAS no conjunto de treino
scaler.fit(X_train)

# 3. Aplicar a transformação nos conjuntos de treino e teste
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDados escalados com sucesso usando StandardScaler (Padronização).")

'''
4. Aplicação da Regressão Logística
A Regressão Logística é um algoritmo de classificação linear (apesar do nome) que modela a probabilidade de um evento pertencer a uma das duas classes (0 ou 1).
'''
# 1. Inicializar o modelo de Regressão Logística
# Definimos random_state para reprodutibilidade
model = LogisticRegression(solver='liblinear', random_state=42)

# 2. Treinar o modelo
model.fit(X_train_scaled, y_train)

print("\nModelo de Regressão Logística treinado com sucesso!")

'''
5. Avaliação do Modelo
Avaliamos o desempenho do modelo no conjunto de teste.
'''
# 1. Fazer previsões no conjunto de teste escalado
y_pred = model.predict(X_test_scaled)

# 2. Avaliar a performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Desempenho do Modelo ---")
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
print("\nRelatório de Classificação (Precision, Recall, F1-Score, Support):")
print(report)

print("\n--- Análise dos Coeficientes ---")
feature_names = X.columns
coefficients = model.coef_[0]

# Criar um DataFrame para facilitar a visualização dos coeficientes
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['Absolute_Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False).drop('Absolute_Coefficient', axis=1)

print("\nCoeficientes do Modelo (Impacto das Features):")
print(coef_df)

'''
Interpretação dos Coeficientes:

Sinal: Um coeficiente positivo (por exemplo, Glucose) significa que um aumento no valor da feature aumenta a probabilidade de o resultado ser 1 (Diabetes). Um coeficiente negativo (BloodPressure) diminui a probabilidade.

Magnitude: Quanto maior o valor absoluto do coeficiente, maior é a importância daquela feature para o modelo. No caso acima, Glucose e BMI (IMC) são geralmente as features mais influentes no diagnóstico de diabetes.
'''
