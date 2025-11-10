# execute_all_fixed.py - Versão corrigida
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

print("🚀 EXECUTANDO PROJETO COMPLETO (VERSÃO CORRIGIDA)...")

# === NOTEBOOK 1 - Limpeza ===
print("\n1️⃣ LIMPEZA DE DADOS...")
df = pd.read_csv('data/healthcare_dataset_v2.csv')

# CORREÇÃO: Converter pressão arterial "130/85" para numérica
def extract_systolic(bp):
    if isinstance(bp, str) and '/' in bp:
        return int(bp.split('/')[0])
    return bp

df['blood_pressure'] = df['blood_pressure'].apply(extract_systolic)

# Tratamento normal
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['smoker'] = df['smoker'].map({'No': 0, 'Yes': 1})
df['readmission_risk'] = df['readmission_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

df.to_csv('data/healthcare_cleaned.csv', index=False)
print("✅ Dados limpos!")

# === NOTEBOOK 2 - Análise ===  
print("\n2️⃣ ANÁLISE EXPLORATÓRIA...")

# Gráfico 1: Distribuição de custos
plt.figure(figsize=(10, 6))
plt.hist(df['medical_costs'], bins=20, alpha=0.7, color='skyblue')
plt.title('Distribuição de Custos Médicos')
plt.xlabel('Custos (R$)')
plt.ylabel('Frequência')
plt.grid(alpha=0.3)
plt.savefig('outputs/custos_histogram.png')
plt.close()

# Gráfico 2: Matriz de correlação (AGORA FUNCIONA!)
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('outputs/correlacao.png')
plt.close()

# Gráfico 3: Satisfação vs Custos
plt.figure(figsize=(10, 6))
plt.scatter(df['satisfaction_score'], df['medical_costs'], alpha=0.6, c=df['age'], cmap='viridis')
plt.colorbar(label='Idade')
plt.title('Satisfação vs Custos (Colorido por Idade)')
plt.xlabel('Score de Satisfação')
plt.ylabel('Custos Médicos (R$)')
plt.grid(alpha=0.3)
plt.savefig('outputs/satisfacao_vs_custos.png')
plt.close()

print("✅ Gráficos gerados!")

# === NOTEBOOK 3 - Modelo ===
print("\n3️⃣ TREINANDO MODELO...")
features = ['age', 'gender', 'bmi', 'blood_pressure', 'cholesterol', 'glucose', 'smoker', 'active_days', 'length_of_stay']
X = df[features]
y = df['medical_costs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Modelo treinado! MAE: R$ {mae:.2f}, R²: {r2:.4f}")

# Salvar modelo
joblib.dump(model, 'outputs/medical_costs_model.pkl')
print("✅ Modelo salvo!")

# Importância das features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Features mais importantes:")
print(feature_importance.head())

print("\n🎉 TUDO PRONTO! Agora volte para o Streamlit e recarregue a página.")