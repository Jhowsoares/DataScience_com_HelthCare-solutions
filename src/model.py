import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class MedicalCostPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepara features para treinamento"""
        self.features = ['age', 'gender', 'bmi', 'blood_pressure', 'cholesterol', 
                        'glucose', 'smoker', 'active_days', 'length_of_stay']
        X = df[self.features]
        y = df['medical_costs']
        return X, y
    
    def train(self, df, test_size=0.2, random_state=42):
        """Treina o modelo Random Forest"""
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Avaliação
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Modelo treinado com sucesso!")
        print(f"MAE: R$ {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return X_test, y_test, y_pred
    
    def predict(self, patient_data):
        """Faz previsão para novos dados"""
        if not self.is_trained:
            raise Exception("Modelo não treinado. Use train() primeiro.")
        
        return self.model.predict(patient_data)
    
    def save_model(self, file_path):
        """Salva o modelo treinado"""
        if self.is_trained:
            joblib.dump(self.model, file_path)
            print(f"Modelo salvo em: {file_path}")
        else:
            print("Nenhum modelo treinado para salvar.")
    
    def load_model(self, file_path):
        """Carrega modelo salvo"""
        self.model = joblib.load(file_path)
        self.is_trained = True
        print(f"Modelo carregado de: {file_path}")

# Exemplo de uso
if __name__ == "__main__":
    from data_processing import DataProcessor
    
    # Carregar e processar dados
    processor = DataProcessor('../data/healthcare_dataset_v2.csv')
    df = processor.load_data()
    df_clean = processor.clean_data()
    
    # Treinar modelo
    predictor = MedicalCostPredictor()
    X_test, y_test, y_pred = predictor.train(df_clean)
    
    # Salvar modelo
    predictor.save_model('../outputs/medical_cost_predictor.pkl')