import os
import joblib
import numpy as np
import pandas as pd
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import MLModel, Prediction, Dataset
from .serializers import MLModelSerializer, PredictionSerializer, DatasetSerializer
from .ml_utils import MLModelManager

# ViewSets for the REST API
class MLModelViewSet(viewsets.ModelViewSet):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer

# Home page view
def home(request):
    return render(request, 'index.html')

# Helper prediction function that loads model and predicts on input_data
def predict_with_model(model_path, input_data):
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        features = model_data['features']

        input_df = pd.DataFrame([input_data])
        input_df = input_df[features]

        # Transform categorical features using saved label encoder
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in label_encoder.classes_:
                input_df[col] = label_encoder.transform(input_df[col])
            else:
                # handle unseen category or fallback (could assign -1 or skip)
                pass

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = float(np.max(probabilities))

        return {
            'success': True,
            'prediction': float(prediction),
            'confidence': confidence
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@api_view(['POST'])
def train_model(request):
    """Train a ML model"""
    try:
        data = request.data
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        model_type = data.get('model_type')  # 'classification' or 'regression'
        algorithm = data.get('algorithm')

        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file_path.path)

        ml_manager = MLModelManager()
        result = ml_manager.train_model(
            df.to_dict('records'),
            target_column,
            model_type,
            algorithm
        )

        if result['success']:
            ml_model = MLModel.objects.create(
                name=f"{algorithm}_{model_type}_{dataset.name}",
                description=f"Trained {algorithm} model on {dataset.name}",
                model_type=model_type
            )

            return JsonResponse({
                'success': True,
                'model_id': ml_model.id,
                'metrics': result['metrics']
            })

        return JsonResponse({'success': False, 'error': result['error']}, status=400)

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@api_view(['POST'])
def make_prediction(request):
    """Make a prediction using a trained model"""
    try:
        data = request.data
        model_id = data.get('model_id')
        input_data = data.get('input_data')

        ml_model = MLModel.objects.get(id=model_id)
        model_path = f"models/{ml_model.model_type}_{ml_model.name.split('_')[0]}.joblib"

        result = predict_with_model(model_path, input_data)

        if result['success']:
            prediction = Prediction.objects.create(
                model=ml_model,
                input_data=input_data,
                prediction_result={'prediction': result['prediction']},
                confidence_score=result['confidence']
            )
            return JsonResponse({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'prediction_id': prediction.id
            })

        return JsonResponse({'success': False, 'error': result['error']}, status=400)

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@api_view(['POST'])
def upload_dataset(request):
    """Upload CSV dataset and save metadata"""
    try:
        file = request.FILES.get('file')
        name = request.data.get('name')
        description = request.data.get('description')

        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)

        df = pd.read_csv(file)
        size = len(df)
        features = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'shape': df.shape
        }

        dataset = Dataset.objects.create(
            name=name,
            description=description,
            file_path=file,
            size=size,
            features=features
        )

        return JsonResponse({
            'success': True,
            'dataset_id': dataset.id,
            'features': features
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@api_view(['GET'])
def get_dataset_preview(request, dataset_id):
    """Return first 10 rows and metadata of dataset"""
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file_path.path)

        preview = {
            'head': df.head(10).to_dict('records'),
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        return JsonResponse(preview)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
