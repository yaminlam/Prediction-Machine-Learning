echo "# 🧠 Prediction Machine Learning API

A Django-based machine learning backend API for uploading datasets, training models, and generating predictions.

## 🚀 Features

- Dataset upload and preview
- Model training (classification & regression)
- Predictions and confidence score
- API endpoints using Django REST Framework

## 🛠️ Tech Stack

- Django 4.x
- Django REST Framework
- scikit-learn
- pandas / numpy
- joblib

## ⚙️ Setup

\`\`\`bash
git clone <your-repo-url>
cd Prediction-Machine-Learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
\`\`\`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/upload/ | Upload CSV dataset |
| POST | /api/train/ | Train model |
| POST | /api/predict/ | Predict using model |
| GET | /api/models/ | List trained models |
| GET | /api/dataset/<id>/preview/ | Preview dataset |

## 📝 Sample Train Payload

\`\`\`json
{
  \"dataset_id\": 1,
  \"target_column\": \"target\",
  \"model_type\": \"classification\",
  \"algorithm\": \"random_forest\"
}
\`\`\`

## 📝 Sample Predict Payload

\`\`\`json
{
  \"model_id\": 1,
  \"input_data\": {
    \"feature1\": 1,
    \"feature2\": \"yes\",
    \"feature3\": 3.5
  }
}
\`\`\`
