from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [path('',views.home,name='home'),
               path('graficos',views.graficos,name='graficos'),
               path('modelo',views.modelo,name='modelo'),
               path('membros',views.membros,name='membros'),
               path('info_modelo',views.info_modelo,name='info_modelo'),
               path('registro',views.registro,name='registro'),
               path('info_tema',views.info_tema,name='info_tema'),
               path('prever',views.prever,name='prever'),
               path('prever/previsao/<int:id>',views.previsao,name='previsao'),
               path('prever/previsao/<int:id>/valor_fornecido',views.valor_fornecido,name='valor_fornecido'),
               path('dados_modelo',views.dados_modelo,name='dados_modelo'),
               path('lista_previsoes',views.lista_previsoes,name='lista_previsoes'),
               path('validacao_modelo',views.validacao_modelo,name='cv_modelo'),
               path('teste_params',views.teste_params,name='teste_params'),
               path('lista_testes',views.lista_testes,name='lista_testes'),]