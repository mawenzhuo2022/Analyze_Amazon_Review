from django.urls import path
from . import views
from .views import UploadCSV, ScoreReview, Pioritize


urlpatterns = [
    path('', views.index, name='index'),
    path('summary', views.summary, name='summary'),
    path('scoring', views.scoring, name='scoring'),
    path('pioritize', views.pioritize, name='pioritize'),
    path('summary/upload/', UploadCSV.as_view(), name='upload_csv'),
    path('scoring/upload/', ScoreReview.as_view(), name='score_review'),
    path('pioritizD/', views.pioritizeD, name='pioritizD'),
    path('pioritizD/upload/', Pioritize.as_view(), name='pioritizeD_csv'),
    
]
