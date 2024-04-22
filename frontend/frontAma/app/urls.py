from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('summary', views.summary, name='summary'),
    path('scoring', views.scoring, name='scoring'),
    path('pioritize', views.pioritize, name='pioritize'),
    
]
