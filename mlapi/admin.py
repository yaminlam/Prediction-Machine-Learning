from django.contrib import admin
from .models import MLModel, Prediction, Dataset

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'created_at', 'is_active')
    list_filter = ('model_type', 'is_active', 'created_at')
    search_fields = ('name', 'description')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('model', 'confidence_score', 'created_at', 'user')
    list_filter = ('model', 'created_at')
    readonly_fields = ('created_at',)

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'size', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('name', 'description')