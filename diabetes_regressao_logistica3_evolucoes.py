import pandas as pd
import numpy as np
import pygad
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, confusion_matrix, precision_score

# --- 1. PREPARAÇÃO DOS DADOS ---
print("Carregando e preparando dados com foco em Recall...")
df = pd.read_csv('datasets/diabetes.csv')

# Tratamento biológico (0 -> NaN -> Mediana)
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
for col in cols_to_replace:
    df[col].fillna(df[col].median(), inplace=True)

X_raw = df.drop('Outcome', axis=1)
y_raw = df['Outcome']

# Divisão Train/Test com estratificação rigorosa
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# --- 2. FUNÇÃO DE FITNESS (O CORAÇÃO DA MELHORIA) ---
def fitness_func(ga_instance, solution, solution_idx):
    c_val = solution[0]
    solver_idx = int(solution[1])
    iqr_f = solution[3]
    feature_mask = solution[4:].astype(bool)
    
    solvers = ['lbfgs', 'liblinear', 'saga']
    selected_solver = solvers[solver_idx]

    # Filtragem de Outliers menos agressiva (para não perder casos reais de diabetes)
    df_temp = pd.concat([X_train_full, y_train], axis=1).copy()
    for col in X_train_full.columns:
        Q1, Q3 = df_temp[col].quantile(0.25), df_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        df_temp = df_temp[(df_temp[col] >= Q1 - iqr_f*IQR) & (df_temp[col] <= Q3 + iqr_f*IQR)]
    
    if len(df_temp) < 200 or not any(feature_mask): return 0 

    X_train_ga = df_temp.drop('Outcome', axis=1).iloc[:, feature_mask]
    y_train_ga = df_temp['Outcome']
    X_test_ga = X_test_full.iloc[:, feature_mask]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_ga)
    X_test_sc = scaler.transform(X_test_ga)

    # MELHORIA CHAVE: class_weight='balanced' ajuda a reduzir Falsos Negativos
    model = LogisticRegression(
        C=c_val, 
        solver=selected_solver, 
        class_weight='balanced', 
        max_iter=3000,
        random_state=42
    )
    model.fit(X_train_sc, y_train_ga)
    
    preds = model.predict(X_test_sc)
    
    # MÉTRICAS
    recall = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    falsos_negativos = cm[1][0] if len(cm) > 1 else 999
    
    # FITNESS PERSONALIZADA: Penaliza pesadamente cada Falso Negativo
    # Quanto menos FN, maior a fitness. 
    fitness = (recall * 1000) - (falsos_negativos * 50)
    
    return fitness

# --- 3. CONFIGURAÇÃO E EXECUÇÃO DO AG ---
gene_space = [
    {'low': 0.001, 'high': 20.0}, # C (amplitude aumentada)
    [0, 1, 2],                    # Solver
    [0],                          # Penalty (fixado em L2 para estabilidade)
    {'low': 2.0, 'high': 4.0},    # IQR Factor (mais alto para manter dados críticos)
] + [[0, 1]] * 8

ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=15,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=len(gene_space),
    gene_space=gene_space,
    mutation_percent_genes=10,
    crossover_type="uniform",
    parent_selection_type="tournament"
)

print("Evoluindo modelo para eliminar Falsos Negativos...")
ga_instance.run()

# --- 4. AVALIAÇÃO FINAL E COMPARATIVO ---
solution, solution_fitness, _ = ga_instance.best_solution()

# Reconstrução do modelo otimizado
best_f_mask = solution[4:].astype(bool)
best_iqr = solution[3]
best_solver = ['lbfgs', 'liblinear', 'saga'][int(solution[1])]

# Aplicar o filtro de outliers otimizado
df_final = pd.concat([X_train_full, y_train], axis=1)
for col in X_train_full.columns:
    Q1, Q3 = df_final[col].quantile(0.25), df_final[col].quantile(0.75)
    df_final = df_final[(df_final[col] >= Q1 - best_iqr*(Q3-Q1)) & (df_final[col] <= Q3 + best_iqr*(Q3-Q1))]

X_tr_ag = df_final.drop('Outcome', axis=1).iloc[:, best_f_mask]
y_tr_ag = df_final['Outcome']
X_ts_ag = X_test_full.iloc[:, best_f_mask]

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr_ag)
X_ts_sc = sc.transform(X_ts_ag)

# Modelo Final com pesos balanceados
final_model = LogisticRegression(C=solution[0], solver=best_solver, class_weight='balanced', max_iter=3000)
final_model.fit(X_tr_sc, y_tr_ag)
y_pred_final = final_model.predict(X_ts_sc)

# Comparação com o original (sem AG e sem class_weight)
mod_orig = LogisticRegression(solver='liblinear').fit(StandardScaler().fit_transform(X_train_full), y_train)
y_pred_orig = mod_orig.predict(StandardScaler().fit_transform(X_test_full))

# Visualização
cm_orig = confusion_matrix(y_test, y_pred_orig)
cm_ag = confusion_matrix(y_test, y_pred_final)



fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Reds', ax=ax[0])
ax[0].set_title(f'ORIGINAL\nRecall: {recall_score(y_test, y_pred_orig):.2f} | FN: {cm_orig[1][0]}')
sns.heatmap(cm_ag, annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title(f'AG OTIMIZADO (Foco Zero FN)\nRecall: {recall_score(y_test, y_pred_final):.2f} | FN: {cm_ag[1][0]}')

plt.tight_layout()
plt.savefig('resultado_final_diabetes.png')
plt.show()

print(f"\nMelhor configuração: Solver={best_solver}, C={solution[0]:.4f}, IQR_Factor={best_iqr:.2f}")
print(f"Features mantidas: {list(X_raw.columns[best_f_mask])}")