import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class MLModelManager:
    def __init__(self):
        self.models = {
            'classification': {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42),
                'svm': SVC(random_state=42, probability=True)
            },
            'regression': {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'svr': SVR()
            }
        }

    def prepare_data(self, data, target_column):
        df = pd.DataFrame(data)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y, scaler, le

    def train_model(self, data, target_column, model_type, algorithm):
        try:
            X, y, scaler, le = self.prepare_data(data, target_column)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = self.models[model_type][algorithm]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if model_type == 'classification':
                metrics = {'accuracy': accuracy_score(y_test, y_pred)}
            else:
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }

            os.makedirs('models', exist_ok=True)
            model_path = f'models/{model_type}_{algorithm}.joblib'
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'label_encoder': le,
                'features': list(pd.DataFrame(data).drop(columns=[target_column]).columns)
            }, model_path)

            return {'success': True, 'metrics': metrics, 'model_path': model_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict(self, model_path, input_data):
        try:
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            label_encoder = model_data['label_encoder']

            input_df = pd.DataFrame([input_data])
            input_df = input_df[features]

            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = label_encoder.transform(input_df[col])

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            confidence = None
            if hasattr(model, 'predict_proba'):
                confidence = float(np.max(model.predict_proba(input_scaled)[0]))

            return {'success': True, 'prediction': float(prediction), 'confidence': confidence}
        except Exception as e:
            return {'success': False, 'error': str(e)}
