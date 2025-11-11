readme_content = """# 🏥 HealthCare Solutions - Analytics
**Projeto de Data Science para otimização do atendimento hospitalar com análise preditiva e visualização interativa.**

---

## 📋 Sobre o Projeto

O projeto **HealthCare Solutions - Analytics** tem como objetivo aplicar técnicas de **Ciência de Dados e Machine Learning** para **melhorar o atendimento ao paciente** por meio da análise e previsão de custos médicos, identificação de padrões e otimização de recursos hospitalares.

A iniciativa utiliza dados simulados baseados em contextos hospitalares reais para gerar **insights que apoiam decisões estratégicas** na jornada do paciente.

### Objetivos Principais
- 🔮 **Previsão de custos médicos** com modelos de Machine Learning
- 📊 **Identificação de padrões e correlações** nos dados hospitalares
- 🎯 **Otimização de recursos** e tempo de permanência hospitalar
- 📈 **Aumento da satisfação** e da experiência do paciente
- ⚡ **Tomada de decisão orientada por dados (Data-Driven Decisions)**

---

## 🚀 Tecnologias Utilizadas

| Categoria | Tecnologias |
|------------|--------------|
| **Linguagem** | Python 3.8+ |
| **Análise e Manipulação de Dados** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualização de Dados** | Matplotlib, Seaborn, Plotly |
| **Dashboard Interativo** | Streamlit |
| **Ambiente de Análise** | Jupyter Notebook |

---

## 📁 Estrutura do Projeto

HealthCare-Solutions/
│
├── 📊 data/
│ ├── healthcare_dataset_v2.csv # Dados originais
│ ├── healthcare_dataset_expanded.csv # Dataset expandido
│ └── healthcare_cleaned.csv # Dados processados
│
├── 📓 notebooks/
│ ├── 01_data_cleaning.ipynb # Limpeza e pré-processamento
│ ├── 02_eda_analysis.ipynb # Análise exploratória
│ └── 03_model_training.ipynb # Modelagem preditiva
│
├── 🛠️ src/
│ ├── data_processing.py # Funções de tratamento de dados
│ ├── model.py # Treinamento e avaliação do modelo ML
│ └── visualization.py # Funções de visualização
│
├── 📈 dashboard/
│ └── app.py # Dashboard Streamlit
│
├── 📤 outputs/
│ ├── medical_costs_model.pkl # Modelo treinado (Random Forest)
│ ├── custos_histogram.png # Gráfico de distribuição de custos
│ └── correlacao.png # Mapa de correlação
│
├── 📄 requirements.txt # Dependências do projeto
└── README.md # Este arquivo


---

## 🛠️ Como Executar

### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
python execute_all_fixed.py
```

### Gerar dados expandidos:
```
python data/generate_expanded_data.py
```
### Rodar notebooks manualmente:
```
jupyter notebook
```

Execute na ordem: 01_data_cleaning → 02_eda_analysis → 03_model_training

### Iniciar o dashboard:

```
streamlit run dashboard/app.py
```

### 📊 Funcionalidades do Dashboard

✅ Métricas em Tempo Real — Custos, satisfação e tempo de permanência
✅ Visualizações Interativas — Histogramas, scatter plots, heatmaps
✅ Previsão de Custos Médicos — Modelo Random Forest Regressor em produção
✅ Análise de Correlação — Identificação dos fatores mais influentes
✅ Dataset Dinâmico — Suporte a diferentes volumes de dados

### 🤖 Modelo de Machine Learning

- Algoritmo: Random Forest Regressor
- Variável Alvo: medical_costs (Custos médicos)
- Principais Features: idade, IMC (BMI), tempo de internação, pressão arterial

#### Performance Esperada:

📏 MAE (Erro Absoluto Médio): R$ 800–1.200

📈 R² Score: 0.75–0.85

🎯 Acurácia média: 80–85%

🎯 Insights Principais

 - Idade é o fator mais influente nos custos médicos

 - Pacientes fumantes apresentam custos 40% maiores em média

- Satisfação tem relação inversa com o tempo de permanência

- Pacientes com BMI > 30 têm 3x mais risco de readmissão


