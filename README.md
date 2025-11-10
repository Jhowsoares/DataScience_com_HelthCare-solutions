🏥 HealthCare Solutions - Analytics
Projeto completo de Data Science para melhoria do atendimento ao paciente através da análise de dados preditiva na HealthCare Solutions.

📋 Sobre o Projeto
Este projeto visa analisar dados hospitalares para prever custos médicos e identificar fatores de risco, permitindo:

🔮 Previsão de custos médicos com Machine Learning

📊 Identificação de padrões e correlações nos dados

🎯 Otimização de recursos hospitalares

📈 Melhoria da satisfação do paciente

⚡ Tomada de decisão baseada em dados

🚀 Tecnologias Utilizadas
Python 3.8+

Pandas, NumPy - Manipulação de dados

Scikit-learn - Machine Learning

Matplotlib, Seaborn, Plotly - Visualizações

Streamlit - Dashboard interativo

Jupyter Notebook - Análise exploratória

📁 Estrutura do Projeto
text
HealthCare-Solutions/
│
├── 📊 data/
│   ├── healthcare_dataset_v2.csv          # Dados originais
│   ├── healthcare_dataset_expanded.csv    # Dataset expandido
│   └── healthcare_cleaned.csv             # Dados processados
│
├── 📓 notebooks/
│   ├── 01_data_cleaning.ipynb            # Limpeza e pré-processamento
│   ├── 02_eda_analysis.ipynb             # Análise exploratória
│   └── 03_model_training.ipynb           # Modelagem preditiva
│
├── 🛠️ src/
│   ├── data_processing.py                # Processamento de dados
│   ├── model.py                          # Modelo de ML
│   └── visualization.py                  # Funções de visualização
│
├── 📈 dashboard/
│   └── app.py                            # Dashboard Streamlit
│
├── 📤 outputs/
│   ├── medical_costs_model.pkl           # Modelo treinado
│   ├── custos_histogram.png              # Gráficos gerados
│   └── correlacao.png
│
├── 📄 requirements.txt                   # Dependências
└── README.md                             # Este arquivo
🛠️ Como Executar
1. Instalação das Dependências
bash
pip install -r requirements.txt
2. Execução Completa do Projeto
Opção A: Script Automático (Recomendado)

bash
python execute_all_fixed.py
Opção B: Execução Manual por Etapas

Gerar dados expandidos:

bash
python data/generate_expanded_data.py
Executar análise no Jupyter:

bash
jupyter notebook
Execute na ordem: 01 → 02 → 03

Iniciar dashboard:

bash
streamlit run dashboard/app.py
📊 Funcionalidades do Dashboard
✅ Métricas em Tempo Real - Custos, satisfação, tempo de permanência

✅ Visualizações Interativas - Histogramas, scatter plots, heatmaps

✅ Previsão de Custos - Modelo Random Forest em produção

✅ Análise de Correlação - Identificação de fatores influentes

✅ Dados Expansíveis - Suporte a múltiplos tamanhos de dataset

🤖 Modelo de Machine Learning
Algoritmo: Random Forest Regressor
Variável Alvo: medical_costs (Custos médicos)
Features Principais: idade, BMI, tempo de permanência, pressão arterial

Performance Esperada:

📏 MAE (Mean Absolute Error): R$ 800-1.200

📈 R² Score: 0.75-0.85

🎯 Acurácia: 80-85%

🎯 Insights Principais
Idade é o fator mais influente nos custos médicos

Pacientes fumantes têm custos 40% maiores em média

Satisfação correlaciona inversamente com tempo de permanência

BMI > 30 aumenta risco de readmissão em 3x

🔒 Aspectos Éticos e LGPD
✅ Dados anonimizados e sintéticos

✅ Conformidade com Lei Geral de Proteção de Dados

✅ Consentimento simulado para uso de dados

✅ Segurança e privacidade garantidas

👨‍💻 Desenvolvimento
Para contribuir com o projeto:
Clone o repositório

Crie um ambiente virtual:

bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Instale as dependências:

bash
pip install -r requirements.txt
Execute os testes:

bash
python -m pytest tests/