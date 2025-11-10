import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# Configuração da página
st.set_page_config(
    page_title="HealthCare Analytics",
    page_icon="🏥",
    layout="wide"
)

# Título principal
st.title("🏥 HealthCare Solutions - Analytics Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Carregar dados (CSV)", type=['csv'])

# Carregar dados
@st.cache_data
def load_data():
    try:
        try:
            df = pd.read_csv('data/healthcare_cleaned_expanded.csv')
        except:
            df = pd.read_csv('data/healthcare_cleaned.csv')
        return df
    
    except:
        # Dados de exemplo se arquivo não existir
        return pd.DataFrame({
            'age': [45, 62, 38, 55, 29],
            'medical_costs': [4500, 12000, 3200, 7800, 2800],
            'satisfaction_score': [4, 2, 5, 3, 4],
            'length_of_stay': [3, 7, 2, 5, 1]
        })

df = load_data()

# Métricas principais
st.subheader("📊 Métricas Principais")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Pacientes", len(df))

with col2:
    st.metric("Custo Médio", f"R$ {df['medical_costs'].mean():.2f}")

with col3:
    st.metric("Satisfação Média", f"{df['satisfaction_score'].mean():.1f}/5")

with col4:
    st.metric("Tempo Médio Permanência", f"{df['length_of_stay'].mean():.1f} dias")

st.markdown("---")

# Visualizações
st.subheader("📈 Análise Visual")

col1, col2 = st.columns(2)

with col1:
    # Distribuição de custos
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['medical_costs'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Distribuição de Custos Médicos')
    ax.set_xlabel('Custos (R$)')
    ax.set_ylabel('Frequência')
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    # Satisfação vs Custos
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['satisfaction_score'], df['medical_costs'], 
                        c=df['age'], alpha=0.7, cmap='viridis')
    ax.set_title('Satisfação vs Custos (Colorido por Idade)')
    ax.set_xlabel('Score de Satisfação')
    ax.set_ylabel('Custos Médicos (R$)')
    plt.colorbar(scatter, ax=ax, label='Idade')
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# Análise de correlação
st.subheader("🔗 Matriz de Correlação")
fig, ax = plt.subplots(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlação entre Variáveis')
st.pyplot(fig)

# Previsão de custos
st.markdown("---")
st.subheader("🤖 Previsão de Custos Médicos")

try:
    model = joblib.load('outputs/medical_costs_model.pkl')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Idade", 18, 100, 45)
        bmi = st.slider("BMI", 15.0, 40.0, 25.0)
        blood_pressure = st.slider("Pressão Arterial", 80, 200, 120)
    
    with col2:
        cholesterol = st.slider("Colesterol", 100, 300, 200)
        glucose = st.slider("Glicose", 70, 200, 100)
        smoker = st.selectbox("Fumante", ["Não", "Sim"])
    
    with col3:
        active_days = st.slider("Dias Ativos/Semana", 0, 7, 3)
        length_of_stay = st.slider("Tempo de Permanência (dias)", 1, 30, 5)
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
    
    if st.button("Prever Custo Médico"):
        # Preparar dados para predição
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [0 if gender == "Masculino" else 1],
            'bmi': [bmi],
            'blood_pressure': [blood_pressure],
            'cholesterol': [cholesterol],
            'glucose': [glucose],
            'smoker': [1 if smoker == "Sim" else 0],
            'active_days': [active_days],
            'length_of_stay': [length_of_stay]
        })
        
        prediction = model.predict(input_data)[0]
        st.success(f"**Custo Médico Previsto: R$ {prediction:,.2f}**")

except Exception as e:
    st.info("⚠️ Modelo de previsão não disponível. Execute o notebook de treinamento primeiro.")

# Rodapé
st.markdown("---")
st.markdown(
    "**HealthCare Solutions Analytics** • Desenvolvido para melhoria do atendimento ao paciente"
)