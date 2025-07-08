from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load example dataset
X, y = load_iris(return_X_y=True)

# Train a simple classifier
model = RandomForestClassifier()
model.fit(X, y)

# Ensure the folder exists
model_dir = os.path.join('mlapi', 'ml_models')
os.makedirs(model_dir, exist_ok=True)

# Save model as my_model.pkl
model_path = os.path.join(model_dir, 'my_model.pkl')
joblib.dump(model, model_path)

print(f"Model saved at: {model_path}")
