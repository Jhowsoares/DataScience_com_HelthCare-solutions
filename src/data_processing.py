import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    
    def load_data(self):
        """Carrega os dados do arquivo CSV"""
        self.df = pd.read_csv(self.file_path)
        print(f"Dados carregados: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")
        return self.df
    
    def clean_data(self):
        """Realiza limpeza básica dos dados"""
        # Remover duplicados
        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        print(f"Duplicados removidos: {initial_rows - self.df.shape[0]}")
        
        # Tratar valores nulos
        null_count = self.df.isnull().sum().sum()
        if null_count > 0:
            self.df = self.df.fillna(self.df.median())
            print(f"Valores nulos tratados: {null_count}")
        
        # Codificar variáveis categóricas
        if 'gender' in self.df.columns:
            self.df['gender'] = self.df['gender'].map({'Male': 0, 'Female': 1})
        
        if 'smoker' in self.df.columns:
            self.df['smoker'] = self.df['smoker'].map({'No': 0, 'Yes': 1})
            
        if 'readmission_risk' in self.df.columns:
            self.df['readmission_risk'] = self.df['readmission_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        return self.df
    
    def get_summary(self):
        """Retorna resumo estatístico"""
        if self.df is not None:
            return self.df.describe()
        else:
            print("Dados não carregados. Use load_data() primeiro.")
            return None

# Exemplo de uso
if __name__ == "__main__":
    processor = DataProcessor('../data/healthcare_dataset_v2.csv')
    df = processor.load_data()
    df_clean = processor.clean_data()
    print(processor.get_summary())