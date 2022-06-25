from django.contrib.auth.models import User
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
class Previsao(models.Model):
    X=models.FloatField()
    Y_prev=models.FloatField(default=None,blank=True,null=True)
    Y_real=models.FloatField(default=None,blank=True,null=True)

class DadosModelo(models.Model):
    X=models.FloatField()
    Y=models.FloatField()

class TesteParams(models.Model):
    Algoritmo=models.CharField(max_length=30)
    Parametro=models.CharField(max_length=10)
    Valores=models.FloatField()
    R2Treino=models.FloatField()
    R2Teste=models.FloatField()
