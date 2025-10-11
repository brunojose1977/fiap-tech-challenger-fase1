import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1. Carregamento dos Dados (Exemplo)
# ----------------------------------------------------------------------
# Assumindo que você já tem um DataFrame 'df' carregado
# Criando um DataFrame de exemplo para demonstração
data = {
    'Coluna_Numérica_Exemplo': [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 150, 200], # Valores 150 e 200 são outliers
    'Coluna_Categórica': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A'],
    'Outra_Coluna': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}
df = pd.DataFrame(data)

print("DataFrame Original:")
print(df)
print("-" * 50)

# ----------------------------------------------------------------------
# 2. SEU CÓDIGO EXISTENTE DE TRATAMENTO DE DADOS (Exemplos)
# ----------------------------------------------------------------------

# Exemplo de tratamento de valores ausentes (imputação com mediana)
df['Coluna_Numérica_Exemplo'].fillna(df['Coluna_Numérica_Exemplo'].median(), inplace=True)

# Exemplo de codificação de variáveis categóricas (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Coluna_Categórica'], drop_first=True)

# ----------------------------------------------------------------------
# 3. NOVO CÓDIGO: TRATAMENTO DE OUTLIERS USANDO O MÉTODO IQR
# ----------------------------------------------------------------------

# 3.1. Definir a coluna para aplicação do IQR
coluna_para_iqr = 'Coluna_Numérica_Exemplo'

# 3.2. Calcular Q1 (25º percentil), Q3 (75º percentil) e IQR
Q1 = df[coluna_para_iqr].quantile(0.25)
Q3 = df[coluna_para_iqr].quantile(0.75)
IQR = Q3 - Q1

# 3.3. Definir os limites de detecção de outliers (Regra de 1.5 * IQR)
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"Estatísticas da coluna '{coluna_para_iqr}':")
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Limite Inferior: {limite_inferior}, Limite Superior: {limite_superior}")
print("-" * 50)

# 3.4. Estratégia de Tratamento (Escolha uma das opções abaixo)

# OPÇÃO A: Remoção (Filtragem) dos Outliers
df_filtrado = df[(df[coluna_para_iqr] >= limite_inferior) & (df[coluna_para_iqr] <= limite_superior)]
print("DataFrame APÓS REMOÇÃO de Outliers:")
print(df_filtrado)

# OPÇÃO B: Capping (Substituição pelos Limites) dos Outliers (Recomendado para manter o tamanho do dataset)
df[coluna_para_iqr] = np.where(
    df[coluna_para_iqr] > limite_superior,
    limite_superior,
    df[coluna_para_iqr]
)

df[coluna_para_iqr] = np.where(
    df[coluna_para_iqr] < limite_inferior,
    limite_inferior,
    df[coluna_para_iqr]
)

print("DataFrame APÓS CAPPING de Outliers:")
print(df)
print("-" * 50)

# ----------------------------------------------------------------------
# 4. Outros Códigos de Pré-processamento/Feature Engineering (Se houver)
# ----------------------------------------------------------------------
# Exemplo de normalização após tratamento de outliers
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df[coluna_para_iqr] = scaler.fit_transform(df[[coluna_para_iqr]])