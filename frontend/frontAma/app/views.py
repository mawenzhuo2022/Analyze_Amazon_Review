from django.http import HttpRequest, HttpResponse
from django.template import loader
# Create your views here.
def index(_: HttpRequest):
    template = loader.get_template('../templates/index.html')
    return HttpResponse(template.render())

def summary(_: HttpRequest):
    template = loader.get_template('../templates/summary.html')
    return HttpResponse(template.render())

def scoring(_: HttpRequest):
    template = loader.get_template('../templates/scoring.html')
    return HttpResponse(template.render())

def pioritize(_: HttpRequest):
    template = loader.get_template('../templates/pioritize.html')
    return HttpResponse(template.render())
