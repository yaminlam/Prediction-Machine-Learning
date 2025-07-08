from django.db import models
from django.contrib.auth.models import User
import json

class MLModel(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    model_type = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.name

class Prediction(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    input_data = models.JSONField()
    prediction_result = models.JSONField()
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    def __str__(self):
        return f"Prediction for {self.model.name} at {self.created_at}"

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    file_path = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    size = models.IntegerField()  # Number of rows
    features = models.JSONField()  # Store feature names and types
    
    def __str__(self):
        return self.name