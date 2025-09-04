import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TravelPricePredictor:
    """
    Modello ML per predizione dinamica prezzi trasporti
    
    CONCETTI ICon IMPLEMENTATI:
    1. Supervised Learning: Regressione per predizioni continue
    2. Feature Engineering: Trasformazione dati dominio → features ML  
    3. Ensemble Methods: Random Forest per non-linearità
    4. Model Evaluation: Cross-validation e metriche multiple
    5. Integration: Connessione con algoritmi ricerca
    """
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.scalers = {}
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature Engineering per predizione prezzi
        
        CONCETTI ICon:
        - Domain Knowledge → Features: geografia, tempo, utente, domanda
        - Categorical Encoding: trasformazione variabili categoriche
        - Feature Selection: mantenere solo features rilevanti
        """
        
        print("[FEATURES] Preparando features per price prediction...")
        
        # Target variable
        target = df['actual_price'].copy()
        
        # Feature DataFrame
        features = df.copy()
        
        # 1. GEOGRAPHIC FEATURES (numeriche già pronte)
        numeric_features = [
            'distance', 'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'
        ]
        
        # 2. TEMPORAL FEATURES 
        temporal_features = [
            'day_of_week', 'hour', 'is_weekend', 'is_peak_hour'
        ]
        
        # 3. USER FEATURES
        user_features = [
            'user_age', 'user_income', 'price_sensitivity', 
            'time_priority', 'comfort_priority'
        ]
        
        # 4. TRANSPORT FEATURES (booleane)
        transport_features = [
            'available_train', 'available_bus', 'available_flight'
        ]
        
        # 5. ECONOMIC/CONTEXTUAL FEATURES
        context_features = [
            'base_demand', 'seasonal_multiplier', 'weather_factor', 'special_event'
        ]
        
        # 6. CATEGORICAL ENCODING
        categorical_features = ['season', 'user_profile', 'transport_type', 'origin', 'destination']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                features[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(features[feature])
            else:
                features[f'{feature}_encoded'] = self.label_encoders[feature].transform(features[feature])
        
        # 7. FEATURE SELECTION
        selected_features = (numeric_features + temporal_features + user_features + 
                           transport_features + context_features + 
                           [f'{cat}_encoded' for cat in categorical_features])
        
        X = features[selected_features]
        self.feature_names = selected_features
        
        print(f"   Features selezionate: {len(selected_features)}")
        print(f"   Shape finale: {X.shape}")
        
        return X, target
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Training e valutazione di multiple modelli
        
        CONCETTI ICon:
        - Cross-Validation: valutazione robusta con K-fold
        - Model Comparison: confronto algoritmi diversi
        - Performance Metrics: MSE, MAE, R² per regressione
        """
        
        print("[TRAINING] Training multiple regression models...")
        
        # Split per test finale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n   Training {model_name}...")
            
            # Scaling per modelli che lo richiedono
            if model_name in ['linear', 'ridge']:
                if model_name not in self.scalers:
                    self.scalers[model_name] = StandardScaler()
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Training
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation (ICon requirement: multiple runs)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_r2_scores = cross_val_score(model, X_train_scaled, y_train,
                                         cv=5, scoring='r2')
            
            # Test set evaluation
            y_pred = model.predict(X_test_scaled)
            
            # Metriche complete
            test_mse = mean_squared_error(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'cv_mse_mean': -cv_scores.mean(),
                'cv_mse_std': cv_scores.std(),
                'cv_r2_mean': cv_r2_scores.mean(),
                'cv_r2_std': cv_r2_scores.std(),
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'model': model
            }
            
            print(f"      CV R²: {cv_r2_scores.mean():.3f} ± {cv_r2_scores.std():.3f}")
            print(f"      Test R²: {test_r2:.3f}, MAE: €{test_mae:.2f}")
        
        # Seleziona miglior modello (highest R²)
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n[BEST MODEL] {best_model_name} con R² = {results[best_model_name]['test_r2']:.3f}")
        
        return results
    
    def predict_price(self, 
                     distance: float,
                     origin: str, 
                     destination: str,
                     transport_type: str,
                     season: str = 'summer',
                     user_profile: str = 'leisure',
                     is_weekend: bool = False,
                     base_demand: float = 1.0,
                     **kwargs) -> float:
        """
        Predice prezzo per singolo viaggio
        
        INTEGRAZIONE con AdvancedPathfinder:
        - Chiamato durante A* search per calcolo g_score
        - Fornisce stime dinamiche invece di valori statici
        """
        
        if self.best_model is None:
            raise ValueError("Modello non ancora trainato! Usa train_models() prima.")
        
        # Costruisci feature vector per predizione
        features = self._build_feature_vector(
            distance=distance,
            origin=origin,
            destination=destination, 
            transport_type=transport_type,
            season=season,
            user_profile=user_profile,
            is_weekend=is_weekend,
            base_demand=base_demand,
            **kwargs
        )
        
        # Scaling se necessario
        if self.best_model_name in self.scalers:
            features_scaled = self.scalers[self.best_model_name].transform([features])
        else:
            features_scaled = [features]
        
        # Predizione
        predicted_price = self.best_model.predict(features_scaled)[0]
        
        return max(predicted_price, 5.0)  # Prezzo minimo €5
    
    def _build_feature_vector(self, **params) -> List[float]:
        """Costruisce feature vector per predizione singola"""
        
        # Default values
        defaults = {
            'distance': params.get('distance', 0),
            'origin_lat': 41.9, 'origin_lon': 12.5,  # Default Roma
            'dest_lat': 45.4, 'dest_lon': 9.2,       # Default Milano
            'day_of_week': 2, 'hour': 10,
            'is_weekend': params.get('is_weekend', False),
            'is_peak_hour': False,
            'user_age': 35, 'user_income': 35000,
            'price_sensitivity': 0.6, 'time_priority': 0.5, 'comfort_priority': 0.5,
            'available_train': True, 'available_bus': True, 'available_flight': False,
            'base_demand': params.get('base_demand', 1.0),
            'seasonal_multiplier': 1.1, 'weather_factor': 1.0, 'special_event': 0,
        }
        
        # Categorical encodings
        season = params.get('season', 'summer')
        user_profile = params.get('user_profile', 'leisure') 
        transport_type = params.get('transport_type', 'train')
        origin = params.get('origin', 'Roma')
        destination = params.get('destination', 'Milano')
        
        # Encode categoricals
        try:
            season_encoded = self.label_encoders['season'].transform([season])[0]
            profile_encoded = self.label_encoders['user_profile'].transform([user_profile])[0]
            transport_encoded = self.label_encoders['transport_type'].transform([transport_type])[0]
            origin_encoded = self.label_encoders['origin'].transform([origin])[0]
            dest_encoded = self.label_encoders['destination'].transform([destination])[0]
        except:
            # Fallback se categoria non vista durante training
            season_encoded = 0
            profile_encoded = 1
            transport_encoded = 0
            origin_encoded = 0
            dest_encoded = 1
        
        # Assembla feature vector nell'ordine atteso
        features = []
        
        # Numeric features (ordine da self.feature_names)
        for feature_name in self.feature_names:
            if feature_name.endswith('_encoded'):
                # Categorical encoded
                if 'season' in feature_name:
                    features.append(season_encoded)
                elif 'user_profile' in feature_name:
                    features.append(profile_encoded)
                elif 'transport_type' in feature_name:
                    features.append(transport_encoded)
                elif 'origin' in feature_name:
                    features.append(origin_encoded)
                elif 'destination' in feature_name:
                    features.append(dest_encoded)
            else:
                # Numeric feature
                features.append(defaults.get(feature_name, 0))
        
        return features
    
    def save_model(self, filepath: str):
        """Salva modello trainato"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"[SAVED] Modello salvato in: {filepath}")
    
    def load_model(self, filepath: str):
        """Carica modello pre-trainato"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name'] 
        self.scalers = model_data['scalers']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"[LOADED] Modello caricato da: {filepath}")

class TravelTimeEstimator:
    """
    Modello ML per stima tempi di viaggio realistici
    Integra fattori ambientali che algoritmi base non considerano
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Feature engineering per time estimation"""
        
        target = df['actual_time'].copy()
        
        # Features rilevanti per tempo
        features = df[['distance', 'transport_type', 'weather_factor', 
                      'is_peak_hour', 'day_of_week', 'special_event']].copy()
        
        # Encode transport type
        if 'transport_type' not in self.label_encoders:
            self.label_encoders['transport_type'] = LabelEncoder()
        
        features['transport_encoded'] = self.label_encoders['transport_type'].fit_transform(
            features['transport_type']
        )
        
        # Drop original categorical
        features = features.drop('transport_type', axis=1)
        
        return features, target
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Training time estimator"""
        
        print("[TRAINING] Time Estimator...")
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        result = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
        self.is_trained = True
        print(f"   Time Estimator R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return result
    
    def predict_time(self, distance: float, transport_type: str, 
                    weather_factor: float = 1.0, is_peak_hour: bool = False,
                    day_of_week: int = 2, special_event: int = 0) -> float:
        """Predice tempo di viaggio"""
        
        if not self.is_trained:
            # Fallback a calcolo base
            base_speeds = {'train': 120, 'bus': 80, 'flight': 500}
            return distance / base_speeds.get(transport_type, 60)
        
        # Encode transport
        try:
            transport_encoded = self.label_encoders['transport_type'].transform([transport_type])[0]
        except:
            transport_encoded = 0
        
        # Feature vector
        features = [[distance, weather_factor, is_peak_hour, 
                    day_of_week, special_event, transport_encoded]]
        
        features_scaled = self.scaler.transform(features)
        predicted_time = self.model.predict(features_scaled)[0]
        
        return max(predicted_time, 0.1)  # Minimo 6 minuti

# Test dei modelli
if __name__ == "__main__":
    from dataset_generator import TravelDatasetGenerator, CityGraph
    
    print("=== TRAVEL PRICE PREDICTOR TRAINING ===")
    
    # Genera dataset
    city_graph = CityGraph()
    generator = TravelDatasetGenerator(city_graph)
    df = generator.generate_travel_scenarios(n_scenarios=500)
    
    # Train Price Predictor
    price_predictor = TravelPricePredictor()
    
    X_price, y_price = price_predictor.prepare_features(df)
    price_results = price_predictor.train_models(X_price, y_price)
    
    # Risultati Price Prediction
    print(f"\n=== PRICE PREDICTION RESULTS ===")
    for model_name, metrics in price_results.items():
        print(f"{model_name:15}: R² = {metrics['test_r2']:.3f}, MAE = €{metrics['test_mae']:.2f}")
    
    # Train Time Estimator  
    time_estimator = TravelTimeEstimator()
    X_time, y_time = time_estimator.prepare_features(df)
    time_results = time_estimator.train(X_time, y_time)
    
    # Test predizioni
    print(f"\n=== PREDICTION TESTS ===")
    
    test_cases = [
        ('Milano', 'Roma', 'train', 'summer', 'business'),
        ('Venezia', 'Napoli', 'flight', 'winter', 'budget'),
        ('Torino', 'Bari', 'bus', 'spring', 'leisure')
    ]
    
    for origin, dest, transport, season, profile in test_cases:
        distance = city_graph._calculate_real_distance(origin, dest)
        
        predicted_price = price_predictor.predict_price(
            distance=distance, origin=origin, destination=dest,
            transport_type=transport, season=season, user_profile=profile
        )
        
        predicted_time = time_estimator.predict_time(
            distance=distance, transport_type=transport
        )
        
        print(f"{origin} → {dest} ({transport}, {profile}):")
        print(f"   Distance: {distance:.0f}km, Price: €{predicted_price:.2f}, Time: {predicted_time:.1f}h")
    
    print(f"\n[DONE] Modelli ML trainati e testati!")