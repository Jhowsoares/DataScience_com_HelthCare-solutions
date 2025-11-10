import pandas as pd
import numpy as np
import random

def expand_existing_data(original_file, target_size=1000):
    """Expande dataset existente mantendo padrões"""
    
    # Carregar dados originais
    df_original = pd.read_csv(original_file)
    
    # Lista para novos dados
    expanded_data = []
    
    # Replicar e variar os dados existentes
    for _ in range(target_size):
        # Selecionar uma linha aleatória como base
        base_row = df_original.iloc[random.randint(0, len(df_original)-1)].copy()
        
        # Adicionar variações realistas
        base_row['patient_id'] = len(expanded_data) + 1
        base_row['age'] = max(18, min(85, base_row['age'] + random.randint(-5, 5)))
        base_row['bmi'] = max(18.5, min(40, base_row['bmi'] + random.uniform(-2, 2)))
        base_row['cholesterol'] = max(100, base_row['cholesterol'] + random.randint(-20, 20))
        base_row['glucose'] = max(70, base_row['glucose'] + random.randint(-10, 10))
        
        # Variação nos custos baseada nas outras variáveis
        cost_variation = random.uniform(0.8, 1.2)
        base_row['medical_costs'] = int(base_row['medical_costs'] * cost_variation)
        
        # Pequena variação na satisfação
        base_row['satisfaction_score'] = max(1, min(5, 
            base_row['satisfaction_score'] + random.randint(-1, 1)))
        
        expanded_data.append(base_row)
    
    return pd.DataFrame(expanded_data)

# Executar expansão
print("Expandindo dataset...")
df_expanded = expand_existing_data('data/healthcare_dataset_v2.csv', 1000)

# Salvar
df_expanded.to_csv('data/healthcare_dataset_expanded_simple.csv', index=False)
print(f"Dataset expandido salvo com {len(df_expanded)} registros")

# Estatísticas
print("\nEstatísticas do dataset expandido:")
print(df_expanded[['age', 'medical_costs', 'satisfaction_score', 'length_of_stay']].describe())