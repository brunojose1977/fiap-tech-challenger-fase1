'''
-----------------------
TechChallenger Fase 2
-----------------------

    Trabalho de entrega da Fase 1 da Pós Graduação IA para Devs da FIAP

    Integrantes do grupo
    --------------------

    -Adalberto Ferreira de Albuquerque Neto (RM368178)
        adalbertonet@outlook.com

    - Bruno José e Silva (RM367064)
        brunojose1977@yahoo.com.br

    - Elton de Souza Machado Simão (RM368289)
        tonsoumasi@gmail.com

    - Lucas Varisco Mendes Bezerra (RM368587)
        lucasv.mendes@hotmail.com

-----------------------
Introdução e Objetivos
-----------------------

    Um grande hospital universitário busca implementar um sistema inteligente de suporte ao diagnóstico, capaz de auxiliar médicos e equipes clínicas na análise inicial de exames e no processamento de dados médicos.

    O presente trabalho visa o desenvolvimento e a validação de um modelo de algoritmo preditivo baseado em Machine Learning para auxiliar no diagnóstico de diabetes. 

    O objetivo principal consiste no treinamento do algoritmo com dados históricos e prever, com base em medições de diagnóstico, se uma paciente tem diabetes.

--------------------------
Dataset Utilizado
--------------------------

    O projeto utilizou o Dataset sobre diabetes.

    Fonte:  N. Inst. of Diabetes & Diges. & Kidney Dis.

    Amostras: O dataset é composto por 768 observações de pacientes.

    Este conjunto de dados é de origem do Instituto Nacional de Diabetes e Doenças Digestivas e Renais (National Institute of Diabetes and Digestive and Kidney Diseases). 

Conteúdo
---------

    Em particular, todas as pacientes aqui são do sexo feminino, com pelo menos 21 anos de idade e de ascendência indígena Pima.

    Mais detalhes sobre a tribo indígena Pima podem ser encontrados em: 

    - https://www.britannica.com/topic/Pima-people

    - https://www.kaggle.com/uciml/pima-indians-diabetes-database

Salvando o Modelo Treinado
--------------------------
Para exportar um modelo de Regressão Logística treinado para a Azure, o formato mais comum e compatível é o Pickle ou o Joblib. 
Foi utilizado o modelo de exportação Joblib, que é eficiente para modelos do scikit-learn.
Posteriormente esse modelo será "deployado" na Azure ML.
O Serviço AutoML carregar esses arquivos facilmente em scripts de "score".
A partir daí toda infraestrutura será provisionada para a execução do modelo

Alternativa:
ONNX (Alternativa): Se você precisar de interoperabilidade máxima, considere salvar o modelo no formato ONNX. O AutoML da Azure tem suporte nativo excelente para esse formato.

>>>> ADICIONAR ARQUI OS OBJETIVOS E REQUISITOS ADICIONAIS DO TECHCHALLENGER FASE 2
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

EXECUTAR_TRATAMENTO_OUTLIERS = True

#---------------------------------------------------
# Função para detectar outliers usando o método IQR
#---------------------------------------------------

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

#--------------------------------------------------------------------------
# Função para Eliminar outliers usando o método IQR em múltiplas colunas
#--------------------------------------------------------------------------
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
    
    print(f" Shape do DataFrame Original: {df.shape}")
    print(f" Linhas removidas: {df.shape[0] - df_resultante.shape[0]}")
    print(f" Shape do DataFrame Limpo: {df_resultante.shape}")

    return df_resultante

#---------------------------------------------------------------------------------
# Função para gerar boxplots para visualização de outliers
#---------------------------------------------------------------------------------
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
    print(f"\n Boxplots de outliers salvos como {filename}")
    plt.close(fig) # Fecha a figura para liberar memória

#---------------------------------------
# Função para limpar a tela do console
#---------------------------------------
def limpar_tela():
    """
    Limpa a tela do console de forma multi-plataforma.
    - Usa 'cls' para Windows (os.name == 'nt').
    - Usa 'clear' para Linux e macOS (os.name é 'posix').
    """
    # Verifica o nome do sistema operacional
    if os.name == 'nt':
        # Comando para Windows
        os.system('cls')
    else:
        # Comando para Linux e macOS
        os.system('clear')

#------------------------------------------------
# Função para salvar o modelo em formato pickle
#------------------------------------------------

def salvar_modelo(modelo, nome_arquivo='modelo_logistico.pkl'):
    """
    Salva o modelo treinado em um arquivo binário.
    """
    try:
        # Cria a pasta 'outputs' se não existir (padrão útil na Azure)
        os.makedirs('outputs', exist_ok=True)
        caminho_completo = os.path.join('outputs', nome_arquivo)
        
        joblib.dump(modelo, caminho_completo)
        print(f"Sucesso! Modelo salvo em: {caminho_completo}")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")


#-----------------------------------------------
# 1. Carregamento e Análise Inicial do Dataset
#----------------------------------------------
limpar_tela()
print("\n Carregando o Dataset diabetes.csv...")
df = pd.read_csv('datasets/diabetes.csv')
print("\n Dataset carregado com sucesso!")

# Visualizar as primeiras linhas e informações gerais
print("\n Primeiras 5 linhas do dataset:")
print(df.head())

print("\n Informações gerais (Tipos de dados e contagem de não-nulos):")
print(df.info())

print("\n Estatísticas descritivas (incluindo Mínimo e Máximo):")
print(df.describe())

#-------------------------------------------------
# 2. Avaliando e tratando valores ausentes (NaN)
#-------------------------------------------------
print("\n Avaliando valores ausentes (NaN) no dataset...")

# Colunas onde '0' representa um valor ausente (NaN) para o dataset Pima Indians Diabetes
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Substituir '0' por NaN nas colunas especificadas
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# Contar a existência de NaNs por feature (coluna)
nan_counts = df.isnull().sum()

# Filtrar apenas as colunas que agora contêm NaNs e ordenar por contagem
nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)

# Contagem de NaNs por feature
print("\n" + "="*50)
print(" Contagem de Valores Ausentes (NaN) por Feature:")
if nan_features.empty:
    print(" Nenhuma feature com valores ausentes encontradas após a substituição de '0'.")
else:
    # Usar .to_string() para garantir o print formatado na saída do terminal
    print(nan_features.to_string())
print("="*50)

## Gráficos de histograma para cada feature
print("\n Gráficos de histograma para cada feature:")

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
    print(f"\n Histograma salvo como {histogram_filename}")
else:
    print("\n O histograma não foi gerado pois não há valores ausentes (NaN) nas features analisadas.")

#-------------------------------------------------
# 3. Tratamento de Dados (Limpeza e Imputação)
#-------------------------------------------------
print("\n Iniciando o tratamento de dados (limpeza e imputação)...")

'''
Tratamento de Dados (Limpeza e Imputação)

No conjunto de dados Pima Indians Diabetes (o que é comumente usado com esta estrutura), algumas colunas como Glucose, BloodPressure, SkinThickness, Insulin, e BMI podem ter valores zero, o que biologicamente é impossível ou altamente improvável (exceto talvez em condições extremas ou erro de medição). Estes zeros são frequentemente tratados como valores nulos (NaN).

Identificação de Zeros Improváveis
Valores de 0 são suspeitos e serão substituídos por NaN para que possamos tratá-los corretamente.
'''

# Colunas a serem verificadas para valores 0 que representam nulos
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Substituir 0 por NaN nas colunas selecionadas
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

print("\n Contagem de valores nulos (NaN) após substituição de zeros:")
print(df.isnull().sum())

'''
Tratamento de Valores Nulos (Imputação)
O método mais comum para imputação é usar a mediana (menos sensível a outliers do que a média) dos dados existentes para preencher os valores ausentes (NaN).
'''
print("\n Iniciando a imputação de valores nulos (NaN) com a mediana das respectivas colunas...")

# Imputação de valores nulos
for col in cols_to_replace:
    # Preencher os NaNs com a mediana da respectiva coluna
    df[col].fillna(df[col].median(), inplace=True)

print("\n Contagem de valores nulos (NaN) após imputação:")
print(df.isnull().sum())

#-------------------------------------------------
# 4. Detecção e Tratamento de Outliers
#-------------------------------------------------
'''
Detecção de Outliers
Usaremos o método do Intervalo Interquartil (IQR) para detectar outliers. Este método é eficaz para identificar valores que estão significativamente distantes da maioria dos dados.    
Basicamente vai retornar a lista de colunas onde foram detectados outliers
'''
print("\n Iniciando a detecção de outliers usando o método IQR...")
outliers = []

if EXECUTAR_TRATAMENTO_OUTLIERS:
    outliers = detectar_outliers_iqr(df)
    if outliers:    
        print(f"\n Colunas com outliers detectados usando IQR: {outliers}")  
    else:
        print("\n Nenhum outlier detectado usando IQR.") 


colunas_para_boxplot = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

print("\n Gerando boxplots para visualização de outliers após imputação...")
gerar_boxplots(df, colunas_para_boxplot, "boxplots_outliers_apos_imputacao.png")
print(" Gerada imagem: boxplots_outliers_apos_imputacao.png")


'''
Tratamento de Outliers
Basicamente as colunas que possuem outliers serão tratadas e os dados outliers eliminadas do dataset
'''
print("\n Iniciando o tratamento de outliers (eliminação) usando o método IQR...")

if EXECUTAR_TRATAMENTO_OUTLIERS and len(outliers) > 0:
    df = tratar_e_eliminar_outliers(df, outliers, iqr_factor=1.5)

''' 
Verificar o resultado do tratamento de outliers com boxplots e estatísticas descritivas
'''
print("\n Estatísticas descritivas após o tratamento de outliers:")

outliers = []

if EXECUTAR_TRATAMENTO_OUTLIERS:
    # Nova passada - nova detecção
    outliers = detectar_outliers_iqr(df)
    if outliers:    
        print(f"\n Colunas com outliers detectados usando IQR: {outliers}")  
    else:
        print("\n Nenhum outlier detectado usando IQR.") 

print(f" Shape do DataFrame Limpo outliers tratados: {df.shape}")

print(" Plotando novamente os Boxplots após o tratamento de outliers...")

colunas_para_boxplot = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

gerar_boxplots(df, colunas_para_boxplot, "boxplots_outliers_apos_imputacao_apos_tratamento_outliers.png")

print(" Gerada imagem: boxplots_outliers_apos_imputacao_apos_tratamento_outliers.png")

# Avalianto e plotando a coorelação entre as features com Gráfico Heatmap
print("\n Gráfico de Correlação (Heatmap) das Features:")

# Calcular a matriz de correlação
correlation_matrix = df.corr()

# Configurar o plot
plt.figure(figsize=(10, 8))

# Usar seaborn para o heatmap
sns.heatmap(correlation_matrix, 
            annot=True, # Mostrar os valores de correlação na célula
            fmt=".2f", # Formato de duas casas decimais
            cmap='coolwarm', # Mapa de cores
            linewidths=.5, # Linhas entre as células
            cbar_kws={'label': 'Coeficiente de Correlação'})

plt.title('Heatmap de Correlação das Features do Dataset Diabetes', fontsize=16)
correlation_filename = 'correlation_heatmap.png'
plt.savefig(correlation_filename)
plt.close()
print(f" Gerada imagem do Heatmap de correlação: {correlation_filename}")

#-------------------------------------------------
# 5. Avaliando a escala dos dados
#-------------------------------------------------

'''
Avaliar a escala dos dados

Colunas do Dataset
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
'''
print("\n Avaliando a escala dos dados das features do dataset...")

colunas_dataset = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

for coluna in colunas_dataset:
    if coluna in df.columns:
        # # Criar o gráfico de boxplot
        # print(f'Imprimindo boxplot para a coluna:', coluna)
        # plt.boxplot(df[coluna])
        # plt.title(coluna)
        # plt.ylabel('Valores')
        # plt.show()

        # Exibir estatísticas descritivas
        print("---------------------------------------")
        print(df[coluna].describe())    

'''
Preparação para o Modelo (Separação e Escalamento)
Definição de Features (X) e Target (y)
Separamos as variáveis preditoras (features) do alvo (target).
'''
print("\n Separando Features (X) e Target (y)...")

# Separar Features (X) e Target (y)
X = df.drop('Outcome', axis=1) # Todas as colunas exceto 'Outcome'
y = df['Outcome']              # A coluna 'Outcome'

print(f"\n Dimensão de X (Features): {X.shape}")
print(f" Dimensão de y (Target): {y.shape}")

'''
Divisão em Treino e Teste
É crucial dividir o dataset para treinar o modelo em um subconjunto e testar seu desempenho em dados nunca vistos.
'''
print("\n Dividindo o dataset em conjuntos de treino e teste...")

# Dividir o dataset em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y) # stratify garante que a proporção de 0s e 1s seja mantida

print(f" Dimensão de X_train: {X_train.shape}")
print(f" Dimensão de X_test: {X_test.shape}")

#-------------------------------------------------
# 6. Padronização dos Dados
#-------------------------------------------------
'''
Padronização e Normalização
O Escalamento de Dados é vital para modelos baseados em distância ou otimização, como a Regressão Logística, pois evita que features com grandes ranges dominem a função de custo.

Regras:

Padronização (StandardScaler): Transforma os dados para ter média 0 e desvio padrão 1. É ideal quando a distribuição dos dados se assemelha a uma normal. É a mais comum para Regressão Logística.

Normalização (MinMaxScaler): Transforma os dados para um range fixo (tipicamente 0 a 1). É útil quando você precisa de valores estritamente positivos (por exemplo, para Redes Neurais).

Sugerida: Para Regressão Logística, a Padronização (StandardScaler) é frequentemente a melhor escolha.
'''
print("\n Iniciando o escalamento dos dados (Padronização)...")

# Técnica escolhida: StandardScaler 
scaler = StandardScaler() 

# 2. Treinar o scaler APENAS no conjunto de treino
scaler.fit(X_train)

# 3. Aplicar a transformação nos conjuntos de treino e teste
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n Dados escalados com sucesso usando StandardScaler (Padronização).")

#-------------------------------------------------
# 6.5. Algoritmo Genético para Otimização de Hiperparâmetros
#-------------------------------------------------
'''
Algoritmo Genético para Otimização de Hiperparâmetros
O algoritmo genético será usado para encontrar os melhores hiperparâmetros 
para o modelo de Regressão Logística, otimizando:
- C (inverso da força de regularização)
- penalty (tipo de regularização: 'l1' ou 'l2')
- solver (algoritmo de otimização)
'''

import random
from sklearn.metrics import f1_score

class AlgoritmoGenetico:
    """
    Implementação de um algoritmo genético para otimização de hiperparâmetros.
    """
    
    def __init__(self, populacao_tamanho=20, geracoes=10, taxa_mutacao=0.1, 
                 taxa_crossover=0.7, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Inicializa o algoritmo genético.
        
        Args:
            populacao_tamanho: Tamanho da população
            geracoes: Número de gerações
            taxa_mutacao: Taxa de mutação (0 a 1)
            taxa_crossover: Taxa de crossover (0 a 1)
            X_train, X_test, y_train, y_test: Dados para treino e teste
        """
        self.populacao_tamanho = populacao_tamanho
        self.geracoes = geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.historico_fitness = []
        self.melhor_individuo = None
        self.melhor_fitness = -float('inf')
        
    def criar_individuo(self):
        """
        Cria um indivíduo (cromossomo) com hiperparâmetros aleatórios.
        
        Returns:
            dict: Dicionário com os hiperparâmetros
        """
        # C: inverso da força de regularização (valores logarítmicos: 0.001 a 100)
        C = 10 ** random.uniform(-3, 2)  # 0.001 a 100
        
        # penalty: tipo de regularização
        penalty = random.choice(['l1', 'l2'])
        
        # solver: algoritmo de otimização (depende do penalty)
        if penalty == 'l1':
            solver = 'liblinear'
        else:
            solver = random.choice(['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'])
        
        return {
            'C': C,
            'penalty': penalty,
            'solver': solver
        }
    
    def criar_populacao(self):
        """
        Cria a população inicial.
        
        Returns:
            list: Lista de indivíduos
        """
        return [self.criar_individuo() for _ in range(self.populacao_tamanho)]
    
    def calcular_fitness(self, individuo):
        """
        Calcula o fitness de um indivíduo (quanto maior, melhor).
        Usa F1-score como métrica de fitness.
        
        Args:
            individuo: Dicionário com hiperparâmetros
            
        Returns:
            float: Fitness do indivíduo (F1-score)
        """
        try:
            # Criar e treinar o modelo com os hiperparâmetros do indivíduo
            modelo = LogisticRegression(
                C=individuo['C'],
                penalty=individuo['penalty'],
                solver=individuo['solver'],
                random_state=42,
                max_iter=1000
            )
            
            modelo.fit(self.X_train, self.y_train)
            
            # Fazer previsões no conjunto de teste
            y_pred = modelo.predict(self.X_test)
            
            # Calcular F1-score como fitness
            fitness = f1_score(self.y_test, y_pred)
            
            return fitness
            
        except Exception as e:
            # Se houver erro, retornar fitness muito baixo
            print(f"  Erro ao calcular fitness: {e}")
            return 0.0
    
    def selecionar_pais(self, populacao, fitness_scores):
        """
        Seleciona pais usando método de roleta (proporcional ao fitness).
        
        Args:
            populacao: Lista de indivíduos
            fitness_scores: Lista de scores de fitness
            
        Returns:
            tuple: Par de pais selecionados
        """
        # Normalizar fitness para probabilidades (garantir valores positivos)
        min_fitness = min(fitness_scores)
        fitness_normalizados = [f - min_fitness + 0.01 for f in fitness_scores]
        soma_fitness = sum(fitness_normalizados)
        
        if soma_fitness == 0:
            # Se todos têm fitness 0, seleção aleatória
            return random.sample(populacao, 2)
        
        probabilidades = [f / soma_fitness for f in fitness_normalizados]
        
        # Seleção por roleta
        pai1 = random.choices(populacao, weights=probabilidades)[0]
        pai2 = random.choices(populacao, weights=probabilidades)[0]
        
        return pai1, pai2
    
    def crossover(self, pai1, pai2):
        """
        Realiza crossover entre dois pais para gerar dois filhos.
        
        Args:
            pai1, pai2: Dicionários com hiperparâmetros dos pais
            
        Returns:
            tuple: Par de filhos gerados
        """
        if random.random() > self.taxa_crossover:
            # Sem crossover, retornar cópias dos pais
            return pai1.copy(), pai2.copy()
        
        # Crossover uniforme: cada gene vem de um pai aleatoriamente
        filho1 = {
            'C': pai1['C'] if random.random() < 0.5 else pai2['C'],
            'penalty': pai1['penalty'] if random.random() < 0.5 else pai2['penalty'],
            'solver': pai1['solver'] if random.random() < 0.5 else pai2['solver']
        }
        
        filho2 = {
            'C': pai2['C'] if random.random() < 0.5 else pai1['C'],
            'penalty': pai2['penalty'] if random.random() < 0.5 else pai1['penalty'],
            'solver': pai2['solver'] if random.random() < 0.5 else pai1['solver']
        }
        
        # Ajustar solver se penalty for l1
        if filho1['penalty'] == 'l1':
            filho1['solver'] = 'liblinear'
        if filho2['penalty'] == 'l1':
            filho2['solver'] = 'liblinear'
        
        return filho1, filho2
    
    def mutar(self, individuo):
        """
        Aplica mutação em um indivíduo.
        
        Args:
            individuo: Dicionário com hiperparâmetros
            
        Returns:
            dict: Indivíduo mutado
        """
        individuo_mutado = individuo.copy()
        
        if random.random() < self.taxa_mutacao:
            # Mutação em C (valor logarítmico)
            individuo_mutado['C'] = 10 ** random.uniform(-3, 2)
        
        if random.random() < self.taxa_mutacao:
            # Mutação em penalty
            individuo_mutado['penalty'] = random.choice(['l1', 'l2'])
            
            # Ajustar solver se necessário
            if individuo_mutado['penalty'] == 'l1':
                individuo_mutado['solver'] = 'liblinear'
            else:
                individuo_mutado['solver'] = random.choice(['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'])
        
        if random.random() < self.taxa_mutacao and individuo_mutado['penalty'] != 'l1':
            # Mutação em solver (apenas se penalty não for l1)
            individuo_mutado['solver'] = random.choice(['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'])
        
        return individuo_mutado
    
    def executar(self):
        """
        Executa o algoritmo genético.
        
        Returns:
            dict: Melhor indivíduo encontrado
        """
        print(f"\n{'='*60}")
        print(" Iniciando Algoritmo Genético para Otimização de Hiperparâmetros")
        print(f"{'='*60}")
        print(f" Tamanho da população: {self.populacao_tamanho}")
        print(f" Número de gerações: {self.geracoes}")
        print(f" Taxa de mutação: {self.taxa_mutacao}")
        print(f" Taxa de crossover: {self.taxa_crossover}")
        print(f"{'='*60}\n")
        
        # Criar população inicial
        populacao = self.criar_populacao()
        
        # Executar gerações
        for geracao in range(self.geracoes):
            print(f" Geração {geracao + 1}/{self.geracoes}...")
            
            # Calcular fitness para todos os indivíduos
            fitness_scores = []
            for i, individuo in enumerate(populacao):
                fitness = self.calcular_fitness(individuo)
                fitness_scores.append(fitness)
                
                # Atualizar melhor indivíduo
                if fitness > self.melhor_fitness:
                    self.melhor_fitness = fitness
                    self.melhor_individuo = individuo.copy()
            
            # Armazenar histórico
            fitness_medio = np.mean(fitness_scores)
            fitness_max = max(fitness_scores)
            self.historico_fitness.append({
                'geracao': geracao + 1,
                'medio': fitness_medio,
                'maximo': fitness_max
            })
            
            print(f"  Fitness médio: {fitness_medio:.4f}")
            print(f"  Fitness máximo: {fitness_max:.4f}")
            print(f"  Melhor indivíduo até agora: C={self.melhor_individuo['C']:.4f}, "
                  f"penalty={self.melhor_individuo['penalty']}, solver={self.melhor_individuo['solver']}")
            
            # Criar nova população
            nova_populacao = []
            
            # Elitismo: manter o melhor indivíduo
            melhor_idx = np.argmax(fitness_scores)
            nova_populacao.append(populacao[melhor_idx].copy())
            
            # Gerar resto da população
            while len(nova_populacao) < self.populacao_tamanho:
                # Selecionar pais
                pai1, pai2 = self.selecionar_pais(populacao, fitness_scores)
                
                # Crossover
                filho1, filho2 = self.crossover(pai1, pai2)
                
                # Mutação
                filho1 = self.mutar(filho1)
                filho2 = self.mutar(filho2)
                
                # Adicionar filhos à nova população
                nova_populacao.append(filho1)
                if len(nova_populacao) < self.populacao_tamanho:
                    nova_populacao.append(filho2)
            
            populacao = nova_populacao
        
        print(f"\n{'='*60}")
        print(" Algoritmo Genético Finalizado!")
        print(f"{'='*60}")
        print(f" Melhor Fitness (F1-Score): {self.melhor_fitness:.4f}")
        print(f" Melhores Hiperparâmetros Encontrados:")
        print(f"   C: {self.melhor_individuo['C']:.6f}")
        print(f"   penalty: {self.melhor_individuo['penalty']}")
        print(f"   solver: {self.melhor_individuo['solver']}")
        print(f"{'='*60}\n")
        
        return self.melhor_individuo

# Configuração do algoritmo genético
USAR_ALGORITMO_GENETICO = True  # Flag para habilitar/desabilitar o algoritmo genético
POPULACAO_TAMANHO = 20
GERACOES = 10
TAXA_MUTACAO = 0.15
TAXA_CROSSOVER = 0.7

#-------------------------------------------------
# 7. Aplicação da Regressão Logística com Otimização por Algoritmo Genético
#-------------------------------------------------
'''
Aplicação da Regressão Logística
A Regressão Logística é um algoritmo de classificação linear (apesar do nome) que modela a probabilidade de um evento pertencer a uma das duas classes (0 ou 1).

Otimização de Hiperparâmetros:
- Se USAR_ALGORITMO_GENETICO = True, usa algoritmo genético para encontrar os melhores hiperparâmetros
- Caso contrário, usa valores padrão
'''
print("\n Iniciando a aplicação da Regressão Logística...")

if USAR_ALGORITMO_GENETICO:
    # Usar algoritmo genético para otimização
    print("\n Otimizando hiperparâmetros usando Algoritmo Genético...")
    
    ag = AlgoritmoGenetico(
        populacao_tamanho=POPULACAO_TAMANHO,
        geracoes=GERACOES,
        taxa_mutacao=TAXA_MUTACAO,
        taxa_crossover=TAXA_CROSSOVER,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )
    
    melhor_hiperparametros = ag.executar()
    
    # Criar modelo com os melhores hiperparâmetros encontrados
    print("\n Criando modelo com os melhores hiperparâmetros encontrados...")
    model = LogisticRegression(
        C=melhor_hiperparametros['C'],
        penalty=melhor_hiperparametros['penalty'],
        solver=melhor_hiperparametros['solver'],
        random_state=42,
        max_iter=1000
    )
    
    # Plotar evolução do fitness
    if ag.historico_fitness:
        geracoes_hist = [h['geracao'] for h in ag.historico_fitness]
        fitness_medio_hist = [h['medio'] for h in ag.historico_fitness]
        fitness_max_hist = [h['maximo'] for h in ag.historico_fitness]
        
        plt.figure(figsize=(10, 6))
        plt.plot(geracoes_hist, fitness_medio_hist, 'b-o', label='Fitness Médio', linewidth=2)
        plt.plot(geracoes_hist, fitness_max_hist, 'r-s', label='Fitness Máximo', linewidth=2)
        plt.xlabel('Geração')
        plt.ylabel('Fitness (F1-Score)')
        plt.title('Evolução do Fitness - Algoritmo Genético')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        evolucao_filename = 'evolucao_algoritmo_genetico.png'
        plt.savefig(evolucao_filename)
        plt.close()
        print(f" Gráfico de evolução salvo como: {evolucao_filename}")
    
else:
    # Usar valores padrão
    print("\n Usando hiperparâmetros padrão...")
model = LogisticRegression(solver='liblinear', random_state=42)

# Treinar o modelo
print("\n Treinando o modelo de Regressão Logística...") 
model.fit(X_train_scaled, y_train)
print("\n Modelo de Regressão Logística treinado com sucesso!")

#-------------------------------------------------
# 8. Salvando o Modelo Treinado
#-------------------------------------------------
print("\n Salvando o modelo treinado em formato pickle...")
salvar_modelo(model, 'modelo_logistico.pkl')
print("\n Modelo salvo com sucesso!")

#-------------------------------------------------
# 9. Avaliação do Desempenho do Modelo
#-------------------------------------------------
'''
Avaliação do Modelo
Avaliamos o desempenho do modelo no conjunto de teste.
'''
print("\n Avaliando o desempenho do modelo no conjunto de teste...")

# Fazer previsões no conjunto de teste escalado
print("\n Fazendo previsões no conjunto de teste...")
y_pred = model.predict(X_test_scaled)

# Avaliar a performance
print("\n Calculando métricas de avaliação...")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n --- Desempenho do Modelo: ---")
print(f" Acurácia no conjunto de teste: {accuracy:.4f}")
print("\n Relatório de Classificação (Precision, Recall, F1-Score, Support):")
print(report)

print("\n --- Análise dos Coeficientes ---")
feature_names = X.columns
coefficients = model.coef_[0]

# Criar um DataFrame para facilitar a visualização dos coeficientes
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['Absolute_Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False).drop('Absolute_Coefficient', axis=1)

print("\n Coeficientes do Modelo (Impacto das Features):")
print(coef_df)

print("""\n\n
Interpretação dos Coeficientes:

Sinal: Um coeficiente positivo (por exemplo, Glucose) significa que um aumento no valor da feature aumenta a probabilidade de o resultado ser 1 (Diabetes). Um coeficiente negativo (BloodPressure) diminui a probabilidade.

Magnitude: Quanto maior o valor absoluto do coeficiente, maior é a importância daquela feature para o modelo. No caso acima, Glucose e BMI (IMC) são geralmente as features mais influentes no diagnóstico de diabetes.
""")

## Aplicando a Matriz grafica de confusão
print("\n Gerando a Matriz de Confusão...")

# Geração e Visualização da Matriz de Confusão
# Calculando a Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)

print("\n --- Matriz de Confusão ---")
print(cm)

print("""\n\n
Interpretação da Matriz de Confusão (CM):
[TN FP]
[FN VP]

Onde:
TN (Verdadeiro Negativo): Casos corretamente previstos como NÃO Diabetes (0).
FP (Falso Positivo): Casos incorretamente previstos como Diabetes (1) - Erro Tipo I.
FN (Falso Negativo): Casos incorretamente previstos como NÃO Diabetes (0) - Erro Tipo II.
VP (Verdadeiro Positivo): Casos corretamente previstos como Diabetes (1).
""")

# Plotagem da Matriz de Confusão (Gerando o Gráfico)
plt.figure(figsize=(8, 6))
# Usando seaborn.heatmap para uma visualização clara
sns.heatmap(cm, 
            annot=True,          # Mostrar os valores na célula
            fmt="d",             # Formato decimal (números inteiros)
            cmap="Blues",        # Mapa de cores
            cbar=False,          # Não mostrar a barra de cores
            linecolor='black',   # Cor das linhas de separação
            linewidths=0.5,      # Largura das linhas de separação
            xticklabels=['Não Diabetes (0)', 'Diabetes (1)'], # Rótulos do eixo X (Previsto)
            yticklabels=['Não Diabetes (0)', 'Diabetes (1)']  # Rótulos do eixo Y (Real)
            )

plt.title('Matriz de Confusão da Regressão Logística')
plt.xlabel('Valor Previsto')
plt.ylabel('Valor Real')

# Salvar a imagem do gráfico da Matriz de Confusão
confusion_matrix_filename = 'confusion_matrix_logistic_regression.png'
plt.savefig(confusion_matrix_filename)
plt.close() # Fecha a figura para liberar memória

print(f"\n Gerado a imagem da Matriz de Confusão salvo como {confusion_matrix_filename}")

print("\n Processo concluído com sucesso!")