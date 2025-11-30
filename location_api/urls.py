from django.urls import path
from . import views

urlpatterns = [
    path('identify/', views.identify_location, name='identify_location'),
    path('health/', views.health_check, name='health_check'),
]