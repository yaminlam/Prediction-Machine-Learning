from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'models', views.MLModelViewSet)
router.register(r'predictions', views.PredictionViewSet)
router.register(r'datasets', views.DatasetViewSet)

urlpatterns = [
    path('', views.home, name='home'),
    path('api/', include(router.urls)),
    path('api/train/', views.train_model, name='train_model'),
    path('api/predict/', views.make_prediction, name='make_prediction'),
    path('api/upload/', views.upload_dataset, name='upload_dataset'),
    path('api/dataset/<int:dataset_id>/preview/', views.get_dataset_preview, name='dataset_preview'),
]
