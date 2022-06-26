from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.contrib.auth.forms import UserCreationForm
from . import forms
from . import models
from RegHelper import carregar_pkl,Reg1Atr,carregar_pkl_treinado
import io, base64
import numpy as np
import pandas as pd
import pickle
# Create your views here.
def supuser(user):
    return user.is_superuser

def home(request):
    template=loader.get_template('home.html')
    return HttpResponse(template.render(request=request))

def graficos(request):
    flike1 = io.BytesIO()
    flike2= io.BytesIO()
    lista_exemplos=models.DadosModelo.objects.all().values()
    X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
    y_ig=np.array([ex['Y'] for ex in lista_exemplos])
    ARVORE=Reg1Atr('rarvore')
    ARVORE.norm_treinar_aval(X_ig,y_ig,tts_rs=0)
    fig1=ARVORE.dispersao_modelo('teste')[0]
    fig2=ARVORE.plotar_arvore(no_rotulos=['PMG(%)','Erro','#Exs','PVG(R$/L)'])[0]
    fig1.savefig(flike1,dpi=150,bbox_inches='tight',pad_inches=0.1)
    fig2.savefig(flike2,dpi=150,bbox_inches='tight',pad_inches=0.1)
    b641 = base64.b64encode(flike1.getvalue()).decode()
    b642 = base64.b64encode(flike2.getvalue()).decode()
    context={'disp':b641,'arvore':b642}
    template=loader.get_template('graficos.html')
    return HttpResponse(template.render(context=context,request=request))

def modelo(request):
    template=loader.get_template('modelo.html')
    return HttpResponse(template.render(request=request))

def membros(request):
    template=loader.get_template('membros.html')
    return HttpResponse(template.render(request=request))

def info_modelo(request):
    template=loader.get_template('info_modelo.html')
    return HttpResponse(template.render(request=request))

def info_tema(request):
    template=loader.get_template('info_tema.html')
    return HttpResponse(template.render(request=request))

def registro(request):
  if request.method == "POST":
    form = forms.NovoUsuario(request.POST)
    if form.is_valid():
      user = form.save()
      return HttpResponseRedirect('/')
  form = forms.NovoUsuario()
  context={"register_form":form}
  template= loader.get_template('registration/register.html')
  return HttpResponse(template.render(context,request))

def prever(request):
    lista_exemplos=models.DadosModelo.objects.all().values()
    X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
    y_ig=np.array([ex['Y'] for ex in lista_exemplos])
    arvore=Reg1Atr('rarvore')
    arvore.norm_treinar_aval(X_ig,y_ig,tts_rs=0)
    if request.method=='POST':
        prev_form=forms.PrevisaoForm(request.POST)
        if prev_form.is_valid():
            prev=prev_form.save()
            X_=request.POST['X']
            y_prev=round(arvore.prever(float(X_)),3)
            prev.Y_prev=y_prev
            prev.save()
            return HttpResponseRedirect(f'/prever/previsao/{prev.id}')
    prev_form=forms.PrevisaoForm()
    context={'prev_form':prev_form}
    template=loader.get_template('prever.html')
    return HttpResponse(template.render(context=context,request=request))

def previsao(request,id):
    prev_ids=models.Previsao.objects.values_list('id',flat=True)
    if id not in prev_ids:
        template=loader.get_template('previsao_nao_existe.html')
        return HttpResponse(template.render(request=request))
    prev=models.Previsao.objects.get(id=id)
    if request.method=='POST':
        y_real=request.POST['y_real']
        prev.Y_real=y_real
        prev.save()
        novo_exemplo=models.DadosModelo(X=prev.X,Y=prev.Y_real)
        novo_exemplo.save()
        return HttpResponseRedirect(f'/prever/previsao/{id}/valor_fornecido')
    template=loader.get_template('previsao.html')
    context={'id':prev.id,'y_prev':prev.Y_prev}
    return HttpResponse(template.render(context=context,request=request))

def valor_fornecido(request,id):
    prev_ids=models.Previsao.objects.values_list('id',flat=True)
    if id not in prev_ids:
        template=loader.get_template('previsao_nao_existe.html')
        return HttpResponse(template.render(request=request))
    prev=models.Previsao.objects.get(id=id)
    template=loader.get_template('valor_fornecido.html')
    context={'id':prev.id,'y_real':prev.Y_real,'X':prev.X}
    return HttpResponse(template.render(context=context,request=request))

def dados_modelo(request):
    lista_exemplos=models.DadosModelo.objects.all().values()
    context={'lista_exemplos':lista_exemplos}
    template=loader.get_template('dados_modelo.html')
    return HttpResponse(template.render(context=context,request=request))

@user_passes_test(supuser)
def lista_previsoes(request):
    lista_prevs=models.Previsao.objects.all().values()
    context={'lista_prevs':lista_prevs}
    template=loader.get_template('lista_previsoes.html')
    return HttpResponse(template.render(context=context,request=request))

@user_passes_test(supuser)
def lista_testes(request):
    lista_testes=models.TesteParams.objects.all().values()
    context={'lista_testes':lista_testes}
    template=loader.get_template('lista_testes.html')
    return HttpResponse(template.render(context=context,request=request))

def validacao_modelo(request):
    ARVORE=Reg1Atr('rarvore')
    lista_exemplos=models.DadosModelo.objects.all().values()
    X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
    y_ig=np.array([ex['Y'] for ex in lista_exemplos])
    cv_d=ARVORE.validacao_cruzada(X_ig,y_ig,cv=5)
    cv_m=cv_d['media']
    cv=cv_d['scores']
    ARVORE_=Reg1Atr('rarvore')
    ARVORE_.norm_treinar_aval(X_ig,y_ig)
    r_treino=float(ARVORE_.aval['Treino'].iloc[0])
    r_teste=float(ARVORE_.aval['Teste'].iloc[0])
    context={'cv':cv,'cv_m':cv_m,'r_treino':r_treino,'r_teste':r_teste}
    template=loader.get_template('cv_modelo.html')
    return HttpResponse(template.render(context=context,request=request))

def teste_params(request):
    if request.method=='POST':
      ARVORE=Reg1Atr('rarvore')
      lista_exemplos=models.DadosModelo.objects.all().values()
      lista_testes=models.TesteParams.objects.all().values()
      X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
      y_ig=np.array([ex['Y'] for ex in lista_exemplos])
      param=request.POST['parametro']
      valores=request.POST['valores']
      try:
        valores_f=[float(i) for i in valores.replace(']','').replace('[','').split(',')]
        res=ARVORE.teste_val_param(param,valores_f,X_ig,y_ig,tts_rs=0)[1]
        res_treino=res['Treino'].to_numpy().ravel()
        res_teste=res['Teste'].to_numpy().ravel()
        for v,rt,rtt in zip(valores_f,res_treino,res_teste):
            t=models.TesteParams(Algoritmo='rarvore',Valores=v,Parametro=param,R2Treino=round(rt,2),R2Teste=round(rtt,2))
            t.save()

        fbytes = io.BytesIO()
        fig=ARVORE.teste_val_param(param,valores_f,X_ig,y_ig,tts_rs=0,plotar=True,subplots_kwds={'figsize':(8,3)})[0]
        fig.savefig(fbytes,dpi=150,bbox_inches='tight',pad_inches=0.1)
        b64 = base64.b64encode(fbytes.getvalue()).decode()
        context={'param':param,'fig':b64}
        template=loader.get_template('teste_params_feito.html')
        return HttpResponse(template.render(context=context,request=request))
      except:
        template=loader.get_template('valores_errados.html')
        return HttpResponse(template.render(request=request))


    template=loader.get_template('teste_params.html')
    return HttpResponse(template.render(request=request))

def outros_modelos(request):
    if request.method=='POST':
        alg=request.POST['algoritmo']
        if alg=='rlmq':
            alg_name='Reg. Linear Mín. Quadrados'
        elif alg=='rlridge':
            alg_name='Reg. Linear Ridge'
        elif alg=='rllasso':
            alg_name='Reg. Linear Lasso'
        elif alg=='lsvm':
            alg_name='Máq. Vetores Suporte Linear'
        elif alg=='rbfsvm':
            alg_name='Máq. Vetores Suporte RBF'
        if request.POST['normalizar']=='nao':
            norm=None
        else:
            norm='minmax'
        lista_exemplos=models.DadosModelo.objects.all().values()
        X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
        y_ig=np.array([ex['Y'] for ex in lista_exemplos])
        if alg=='rlridge' or alg=='rllasso':
            modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'alpha':0.0001},normalizador=norm)
        elif alg=='lsvm' or alg=='rbfsvm':
            modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'C':1},normalizador=norm)
        else:
            modelo=Reg1Atr(algoritmo=alg,normalizador=norm)
        modelo.norm_treinar_aval(X_ig,y_ig,tts_rs=0)
        fbytes = io.BytesIO()
        fig=modelo.dispersao_modelo('teste')[0]
        fig.savefig(fbytes,dpi=150,bbox_inches='tight',pad_inches=0.1)
        b64 = base64.b64encode(fbytes.getvalue()).decode()
        r_treino=float(modelo.aval['Treino'].iloc[0])
        r_teste=float(modelo.aval['Teste'].iloc[0])
        context={'alg_nome':alg_name,'disp':b64,'alg':alg}
        template=loader.get_template('outro_modelo_treinado.html')
        return HttpResponse(template.render(context=context,request=request))
    template=loader.get_template('outros_modelos.html')
    return HttpResponse(template.render(request=request))

def prever_outro(request,alg):
    if alg=='rlmq':
        alg_name='Reg. Linear Mín. Quadrados'
    elif alg=='rlridge':
        alg_name='Reg. Linear Ridge'
    elif alg=='rllasso':
        alg_name='Reg. Linear Lasso'
    elif alg=='lsvm':
        alg_name='Máq. Vetores Suporte Linear'
    elif alg=='rbfsvm':
        alg_name='Máq. Vetores Suporte RBF'
    if request.method=='POST':
        lista_exemplos=models.DadosModelo.objects.all().values()
        X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
        y_ig=np.array([ex['Y'] for ex in lista_exemplos])
        if alg=='rlridge' or alg=='rllasso':
            modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'alpha':0.0001},normalizador=None)
        elif alg=='lsvm' or alg=='rbfsvm':
            modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'C':1},normalizador=None)
        else:
            modelo=Reg1Atr(algoritmo=alg,normalizador=None)
        modelo.norm_treinar_aval(X_ig,y_ig,tts_rs=0)
        X=request.POST['X']
        y_prev=round(modelo.prever(float(X)),3)
        context={'alg':alg,'alg_nome':alg_name,'y_prev':y_prev}
        template=loader.get_template('outro_modelo_previsao.html')
        return HttpResponse(template.render(context=context,request=request))
    template=loader.get_template('outro_modelo_prever.html')
    return HttpResponse(template.render(request=request))

def validacao_outro(request,alg):
    if alg=='rlmq':
        alg_name='Reg. Linear Mín. Quadrados'
    elif alg=='rlridge':
        alg_name='Reg. Linear Ridge'
    elif alg=='rllasso':
        alg_name='Reg. Linear Lasso'
    elif alg=='lsvm':
        alg_name='Máq. Vetores Suporte Linear'
    elif alg=='rbfsvm':
        alg_name='Máq. Vetores Suporte RBF'
    lista_exemplos=models.DadosModelo.objects.all().values()
    X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
    y_ig=np.array([ex['Y'] for ex in lista_exemplos])
    if alg=='rlridge' or alg=='rllasso':
        modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'alpha':0.0001},normalizador=None)
        modelo_=Reg1Atr(algoritmo=alg,algoritmo_kwds={'alpha':0.0001},normalizador=None)
    elif alg=='lsvm' or alg=='rbfsvm':
        modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'C':1},normalizador=None)
        modelo_=Reg1Atr(algoritmo=alg,algoritmo_kwds={'C':1},normalizador=None)
    else:
        modelo=Reg1Atr(algoritmo=alg,normalizador=None)
        modelo_=Reg1Atr(algoritmo=alg,normalizador=None)
    cv_d=modelo.validacao_cruzada(X_ig,y_ig,cv=5)
    cv_m=cv_d['media']
    cv=cv_d['scores']
    modelo_.norm_treinar_aval(X_ig,y_ig)
    r_treino=float(modelo_.aval['Treino'].iloc[0])
    r_teste=float(modelo_.aval['Teste'].iloc[0])
    context={'alg_nome':alg_name,'cv':cv,'cv_m':cv_m,'r_treino':r_treino,'r_teste':r_teste}
    template=loader.get_template('outro_modelo_cv.html')
    return HttpResponse(template.render(context=context,request=request))

def teste_params_outro(request,alg):
    if alg=='rlmq':
        alg_name='Reg. Linear Mín. Quadrados'
    elif alg=='rlridge':
        alg_name='Reg. Linear Ridge'
    elif alg=='rllasso':
        alg_name='Reg. Linear Lasso'
    elif alg=='lsvm':
        alg_name='Máq. Vetores Suporte Linear'
    elif alg=='rbfsvm':
        alg_name='Máq. Vetores Suporte RBF'

    if request.method=='POST':
        if alg=='rlridge' or alg=='rllasso':
            modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'alpha':0.0001},normalizador=None)
            modelo_=Reg1Atr(algoritmo=alg,algoritmo_kwds={'alpha':0.0001},normalizador=None)
        elif alg=='lsvm' or alg=='rbfsvm':
            modelo=Reg1Atr(algoritmo=alg,algoritmo_kwds={'C':1},normalizador=None)
            modelo_=Reg1Atr(algoritmo=alg,algoritmo_kwds={'C':1},normalizador=None)
        else:
            modelo=Reg1Atr(algoritmo=alg,normalizador=None)
            modelo_=Reg1Atr(algoritmo=alg,normalizador=None)

        lista_exemplos=models.DadosModelo.objects.all().values()
        X_ig=np.array([ex['X'] for ex in lista_exemplos]).reshape(-1,1)
        y_ig=np.array([ex['Y'] for ex in lista_exemplos])
        param=request.POST['parametro']
        valores=request.POST['valores']
        try:
            valores_f=[float(i) for i in valores.replace(']','').replace('[','').split(',')]
            res=modelo.teste_val_param(param,valores_f,X_ig,y_ig,tts_rs=0)[1]
            res_treino=res['Treino'].to_numpy().ravel()
            res_teste=res['Teste'].to_numpy().ravel()
            for v,rt,rtt in zip(valores_f,res_treino,res_teste):
                t=models.TesteParams(Algoritmo=alg,Valores=v,Parametro=param,R2Treino=round(rt,2),R2Teste=round(rtt,2))
                t.save()
            fbytes = io.BytesIO()
            fig=modelo.teste_val_param(param,valores_f,X_ig,y_ig,tts_rs=0,plotar=True,subplots_kwds={'figsize':(8,3)})[0]
            fig.savefig(fbytes,dpi=150,bbox_inches='tight',pad_inches=0.1)
            b64 = base64.b64encode(fbytes.getvalue()).decode()
            context={'alg_nome':alg_name,'param':param,'fig':b64}
            template=loader.get_template('outro_modelo_teste_params_feito.html')
            return HttpResponse(template.render(context=context,request=request))
        except:
            template=loader.get_template('valores_errados.html')
            return HttpResponse(template.render(request=request))
    context={'alg_nome':alg_name}
    template=loader.get_template('outro_modelo_teste_params.html')
    return HttpResponse(template.render(context=context,request=request))
