from django.urls import path
from . import views
from .views import UploadCSV, ScoreReview, Prioritize


urlpatterns = [
    path('', views.index, name='index'),
    path('summary', views.summary, name='summary'),
    path('scoring', views.scoring, name='scoring'),
    path('prioritize', views.prioritize, name='prioritize'),
    path('summary/upload/', UploadCSV.as_view(), name='upload_csv'),
    path('scoring/upload/', ScoreReview.as_view(), name='score_review'),
    path('prioritizD/', views.prioritizeD, name='prioritizeD'),
    path('prioritizD/upload/', Prioritize.as_view(), name='prioritizeD_csv'),
    
]
