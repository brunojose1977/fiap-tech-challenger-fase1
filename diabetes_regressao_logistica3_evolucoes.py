import pandas as pd
import numpy as np
import pygad
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix

# --- 1. PREPARAÇÃO DOS DADOS ---
print("Carregando dados para maximização de redução de FN...")
df = pd.read_csv('datasets/diabetes.csv')

cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
for col in cols_to_replace:
    df[col].fillna(df[col].median(), inplace=True)

X_raw = df.drop('Outcome', axis=1)
y_raw = df['Outcome']

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# --- 2. FUNÇÃO DE FITNESS (FOCO TOTAL EM MINIMIZAR FN) ---
def fitness_func(ga_instance, solution, solution_idx):
    c_val = solution[0]
    solver_idx = int(solution[1])
    iqr_f = solution[3]
    feature_mask = solution[4:].astype(bool)
    
    solvers = ['lbfgs', 'liblinear', 'saga']
    selected_solver = solvers[solver_idx]

    df_temp = pd.concat([X_train_full, y_train], axis=1).copy()
    for col in X_train_full.columns:
        Q1, Q3 = df_temp[col].quantile(0.25), df_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        df_temp = df_temp[(df_temp[col] >= Q1 - iqr_f*IQR) & (df_temp[col] <= Q3 + iqr_f*IQR)]
    
    if len(df_temp) < 150 or not any(feature_mask): return -99999 

    X_train_ga = df_temp.drop('Outcome', axis=1).iloc[:, feature_mask]
    y_train_ga = df_temp['Outcome']
    X_test_ga = X_test_full.iloc[:, feature_mask]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_ga)
    X_test_sc = scaler.transform(X_test_ga)

    # Note o uso de class_weight='balanced' para forçar o modelo a olhar para a classe minoritária
    model = LogisticRegression(C=c_val, solver=selected_solver, class_weight='balanced', max_iter=3000, random_state=42)
    model.fit(X_train_sc, y_train_ga)
    
    preds = model.predict(X_test_sc)
    cm = confusion_matrix(y_test, preds)
    
    if len(cm) < 2: return -99999
    
    falsos_negativos = cm[1][0]
    recall = recall_score(y_test, preds, zero_division=0)
    
    # ESTRATÉGIA DE FITNESS: 
    # 1. Recompensamos o Recall (escala 0-1000)
    # 2. Penalizamos PESADAMENTE cada FN (500 pontos por erro)
    # Isso força o AG a descartar qualquer solução que deixe passar casos positivos.
    fitness = (recall * 1000) - (falsos_negativos * 500)
    
    return fitness

# --- 3. CALLBACK DE MONITORAMENTO ---
def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_sol, best_fit, _ = ga_instance.best_solution()
    solvers = ['lbfgs', 'liblinear', 'saga']
    print(f"Gen {gen:03d} | Fitness: {best_fit:8.2f} | C: {best_sol[0]:.4f} | Solver: {solvers[int(best_sol[1])]}")

# --- 4. CONFIGURAÇÃO E EXECUÇÃO DO AG ---
gene_space = [{'low': 0.001, 'high': 50.0}, [0, 1, 2], [0], {'low': 1.5, 'high': 5.0}] + [[0, 1]] * 8

ga_instance = pygad.GA(
    num_generations=100, # Aumentado para busca mais exaustiva
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=40,
    num_genes=len(gene_space),
    gene_space=gene_space,
    on_generation=on_generation,
    mutation_percent_genes=15 # Aumentado para evitar estagnação
)

print("\nIniciando busca agressiva por Zero Falsos Negativos...")
ga_instance.run()

# --- 5. GERAÇÃO DOS GRÁFICOS ---

# Plot Fitness
plt.figure(figsize=(10, 5))
ga_instance.plot_fitness(title="Evolução da Fitness (Minimização de FN)")
plt.grid(True, alpha=0.3)
plt.savefig('evolucao_fn_fitness.png', dpi=300)
plt.show()

# --- 6. AVALIAÇÃO FINAL ---
solution, _, _ = ga_instance.best_solution()
best_f_mask = solution[4:].astype(bool)
best_iqr = solution[3]
best_solver = ['lbfgs', 'liblinear', 'saga'][int(solution[1])]

# Reconstrução para Validação
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

final_model = LogisticRegression(C=solution[0], solver=best_solver, class_weight='balanced', max_iter=5000)
final_model.fit(X_tr_sc, y_tr_ag)
y_pred_final = final_model.predict(X_ts_sc)

# Comparativo
cm_ag = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ag, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão Otimizada para FN\nRecall: {recall_score(y_test, y_pred_final):.4f}')
plt.savefig('resultado_final_fn.png', dpi=300)
plt.show()

print(f"\nResultado Final: FN encontrados = {cm_ag[1][0]}")
print(f"Features Selecionadas: {list(X_raw.columns[best_f_mask])}")