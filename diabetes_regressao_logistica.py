'''
Isso é um ótimo projeto! Como especialista em Machine Learning, Python e Ciência de Dados, vou guiar você através das etapas para carregar o conjunto de dados "diabetes.csv", realizar o pré-processamento necessário, e aplicar um modelo de Regressão Logística.

Para este projeto, usaremos as bibliotecas padrão de ciência de dados: pandas para manipulação de dados, numpy para operações numéricas, e scikit-learn (sklearn) para pré-processamento e modelagem.

1. Carregamento e Análise Inicial do Dataset
O primeiro passo é carregar o arquivo e inspecioná-lo. Assumirei que o arquivo diabetes.csv está no mesmo diretório que o seu script Python ou notebook.
'''
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

EXECUTAR_TRATAMENTO_OUTLIERS = True

def detectar_outliers_iqr(df):
    """
    Detecta outliers em cada coluna numérica de um DataFrame usando o método IQR.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.

    Returns:
        list: Uma lista de nomes de colunas onde outliers foram encontrados.
    """
    colunas_com_outliers = []
    
    # Colunas numéricas (assumindo que todas, exceto 'Outcome' que é binária, são numéricas)
    colunas_numericas = df.drop(columns=['Outcome'], errors='ignore').select_dtypes(include=np.number).columns
    
    for coluna in colunas_numericas:
        # Calcular Q1 (25º percentil) e Q3 (75º percentil)
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        
        # Calcular o Intervalo Interquartil (IQR)
        IQR = Q3 - Q1
        
        # Definir os limites inferior e superior para outliers
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Identificar outliers
        # Outlier é um valor abaixo do limite inferior OU acima do limite superior
        outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
        
        # Verificar se há algum outlier
        if not outliers.empty:
            colunas_com_outliers.append(coluna)
            
    return colunas_com_outliers

def tratar_e_eliminar_outliers(df, colunas_com_outliers, iqr_factor=1.5):
    """
    Remove as linhas de um DataFrame que contêm outliers nas colunas especificadas.
    O método de eliminação é o Intervalo Interquartil (IQR).

    Args:
        df (pd.DataFrame): O DataFrame de entrada.
        colunas_com_outliers (list): Lista de nomes de colunas para tratar.
        iqr_factor (float): Fator multiplicativo do IQR (padrão é 1.5).

    Returns:
        pd.DataFrame: Um novo DataFrame sem os outliers nas colunas especificadas.
    """
    df_limpo = df.copy()
    
    # Criamos uma máscara de filtro para manter as linhas
    # Começamos com True, significando que todas as linhas são mantidas inicialmente
    mascara_filtro = pd.Series(True, index=df_limpo.index)
    
    for coluna in colunas_com_outliers:
        if coluna in df_limpo.columns:
            # 1. Calcular Q1 e Q3
            Q1 = df_limpo[coluna].quantile(0.25)
            Q3 = df_limpo[coluna].quantile(0.75)
            
            # 2. Calcular o IQR
            IQR = Q3 - Q1
            
            # 3. Definir os limites inferior e superior
            limite_inferior = Q1 - iqr_factor * IQR
            limite_superior = Q3 + iqr_factor * IQR
            
            # 4. Criar uma máscara para as linhas que *NÃO* são outliers na coluna atual
            # Uma linha NÃO é outlier se o valor for >= limite_inferior E <= limite_superior
            mascara_coluna = (df_limpo[coluna] >= limite_inferior) & (df_limpo[coluna] <= limite_superior)
            
            # 5. Atualizar a máscara de filtro combinada
            # Usamos o operador '&' (AND) para garantir que uma linha seja mantida 
            # *SOMENTE SE* não for outlier em *TODAS* as colunas analisadas até agora.
            mascara_filtro = mascara_filtro & mascara_coluna
            
    # Aplicar a máscara final para obter o DataFrame limpo
    df_resultante = df_limpo[mascara_filtro]
    
    print(f"Shape do DataFrame Original: {df.shape}")
    print(f"Linhas removidas: {df.shape[0] - df_resultante.shape[0]}")
    print(f"Shape do DataFrame Limpo: {df_resultante.shape}")

    return df_resultante

# 1. Carregar o dataset
# try:
#     df = pd.read_csv('/home/brunojose/devops/python/Fiap-TechChallenger1/datasets/diabetes.csv')
#     print("Dataset carregado com sucesso!")
# except FileNotFoundError:
#     print("Erro: O arquivo 'diabetes.csv' não foi encontrado. Certifique-se de que ele está no diretório correto.")
#     # Usar um dataset de exemplo para demonstração se necessário, mas o código abaixo assume o carregamento.
#     # return 
df = pd.read_csv('/home/brunojose/devops/python/Fiap-TechChallenger1/datasets/diabetes.csv')
print("Dataset carregado com sucesso!")

# Visualizar as primeiras linhas e informações gerais
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações gerais (Tipos de dados e contagem de não-nulos):")
print(df.info())

print("\Estatísticas descritivas (incluindo Mínimo e Máximo):")
print(df.describe())

print("\nGráficos de histograma para cada feature:")
## Gráficos de histograma para cada feature

# Colunas onde '0' representa um valor ausente (NaN) para o dataset Pima Indians Diabetes
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Substituir '0' por NaN nas colunas especificadas
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# Contar a existência de NaNs por feature (coluna)
nan_counts = df.isnull().sum()

# Filtrar apenas as colunas que agora contêm NaNs e ordenar por contagem
nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)

# --- Parte de Geração de Saída para o Terminal (Contagem de NaNs) ---
print("\n" + "="*50)
print("Contagem de Valores Ausentes (NaN) por Feature:")
if nan_features.empty:
    print("Nenhuma feature com valores ausentes encontradas após a substituição de '0'.")
else:
    # Usar .to_string() para garantir o print formatado na saída do terminal
    print(nan_features.to_string())
print("="*50)

# --- Parte de Geração do Histograma (Barra) ---

if not nan_features.empty:
    # Criar a figura do histograma (Gráfico de Barras)
    plt.figure(figsize=(10, 6))
    plt.bar(nan_features.index, nan_features.values, color='skyblue')
    plt.title('Histograma de Valores Ausentes (NaN) por Feature')
    plt.xlabel('Feature')
    plt.ylabel('Contagem de Valores Ausentes')
    plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para melhor visualização
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Ajusta automaticamente o layout

    # Salvar a imagem do histograma (necessário para visualização em ambientes que não exibem plots interativos)
    histogram_filename = 'nan_features_histogram.png'
    plt.savefig(histogram_filename)
    print(f"\nHistograma salvo como {histogram_filename}")
else:
    print("\nO histograma não foi gerado pois não há valores ausentes (NaN) nas features analisadas.")


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
Tratamento de Valores Nulos (Imputação)
O método mais comum para imputação é usar a mediana (menos sensível a outliers do que a média) dos dados existentes para preencher os valores ausentes (NaN).
'''

# Imputação de valores nulos
for col in cols_to_replace:
    # Preencher os NaNs com a mediana da respectiva coluna
    df[col].fillna(df[col].median(), inplace=True)

print("\nContagem de valores nulos (NaN) após imputação:")
print(df.isnull().sum())

# primeira passada
outliers = []
'''
Detecção de Outliers
Usaremos o método do Intervalo Interquartil (IQR) para detectar outliers. Este método é eficaz para identificar valores que estão significativamente distantes da maioria dos dados.    
Basicamente vai retornar a lista de colunas onde foram detectados outliers
'''
if EXECUTAR_TRATAMENTO_OUTLIERS:
    outliers = detectar_outliers_iqr(df)
    if outliers:    
        print(f"\nColunas com outliers detectados usando IQR: {outliers}")  
    else:
        print("\nNenhum outlier detectado usando IQR.") 

# --- NOVA FUNÇÃO ADICIONADA: gerar_boxplots ---
def gerar_boxplots(df, colunas, filename='boxplots_outliers_apos_imputacao.png'):
    """
    Gera boxplots para as colunas especificadas para visualizar outliers.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.
        colunas (list): Lista de nomes de colunas para gerar boxplots.
        filename (str): Nome do arquivo para salvar o gráfico.
    """
    n_colunas = len(colunas)
    # Determinar o layout da subfigura (ex: 3 colunas)
    cols = 3
    rows = (n_colunas + cols - 1) // cols
    
    # Aumentar um pouco o figsize para melhor visualização dos 7 gráficos
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Achatar o array de axes para fácil iteração, caso seja multidimensional
    if n_colunas > 1:
        axes = axes.flatten()
    else:
        # Se for apenas uma coluna, axes não é um array
        axes = [axes] 

    for i, coluna in enumerate(colunas):
        if coluna in df.columns:
            ax = axes[i]
            # Criar o boxplot
            # vert=False para orientação horizontal, facilitando a leitura da dispersão
            df.boxplot(column=coluna, ax=ax, vert=False, patch_artist=True) 
            ax.set_title(f'Boxplot de {coluna}')
            ax.set_xlabel('Valor')
        
    # Ocultar subplots vazios, se houver
    for j in range(n_colunas, rows * cols):
        fig.delaxes(axes[j])
        
    plt.suptitle('Visualização de Outliers (Método Boxplot) - Dados Após Imputação', fontsize=16)
    # Ajusta o layout para a suptitle e subtítulos não se sobreporem
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(filename)
    print(f"\nBoxplots de outliers salvos como {filename}")
    plt.close(fig) # Fecha a figura para liberar memória
# --- FIM DA NOVA FUNÇÃO ---

# Primeira passada
# --- INSERÇÃO DA CHAMADA DA NOVA FUNÇÃO ---
colunas_para_boxplot = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
gerar_boxplots(df, colunas_para_boxplot, "boxplots_outliers_apos_imputacao.png")
# --- FIM DA INSERÇÃO ---


'''
Tratamento de Outliers
Basicamente as colunas que possuem outliers serão tratadas e os dados outliers eliminadas do dataset
'''
if EXECUTAR_TRATAMENTO_OUTLIERS and len(outliers) > 0:
    df = tratar_e_eliminar_outliers(df, outliers, iqr_factor=1.5)

''' 
Verificar o resultado do tratamento de outliers com boxplots e estatísticas descritivas
'''

print("\nVerificar o resultado da detecção de outliers depois do tratamento de outliers com boxplots e estatísticas descritivas\n")

# segunda passada - nova detecção de outliers
outliers = []
'''
Detecção de Outliers
Usaremos o método do Intervalo Interquartil (IQR) para detectar outliers. Este método é eficaz para identificar valores que estão significativamente distantes da maioria dos dados.    
Basicamente vai retornar a lista de colunas onde foram detectados outliers
'''
if EXECUTAR_TRATAMENTO_OUTLIERS:
    # Nova passada - nova detecção
    outliers = detectar_outliers_iqr(df)
    if outliers:    
        print(f"\nColunas com outliers detectados usando IQR: {outliers}")  
    else:
        print("\nNenhum outlier detectado usando IQR.") 

print("\n Verificar o resultado do tratamento de outliers com boxplots e estatísticas descritivas\n")

print(f"Shape do DataFrame Limpo outliers tratados: {df.shape}")

# Nova passada
# Plotar novamente os Boxplots após o tratamento de outliers
# --- INSERÇÃO DA CHAMADA DA NOVA FUNÇÃO ---
colunas_para_boxplot = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
gerar_boxplots(df, colunas_para_boxplot, "boxplots_outliers_apos_imputacao_apos_tratamento_outliers.png")
# --- FIM DA INSERÇÃO ---

### >>>>> estou aqui

'''
Avaliar a escala dos dados

Colunas do Dataset
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
'''

colunas_dataset = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

for coluna in colunas_dataset:
    if coluna in df.columns:
        # Criar o gráfico de boxplot
        print('fImprimindo boxplot para a coluna:', coluna)
        plt.boxplot(df[coluna])
        plt.title(coluna)
        plt.ylabel('Valores')
        plt.show()
        # Exibir estatísticas descritivas
        print(df[coluna].describe())    

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

# %%
