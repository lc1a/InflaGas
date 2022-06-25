import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn
import re
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score,KFold

class Reg1Atr():

  def __init__(self,algoritmo,algoritmo_kwds={},normalizador=None):
    if algoritmo=='rlmq':
      self.modelo=LinearRegression(**algoritmo_kwds)
    elif algoritmo=='rlridge':
      self.modelo=Ridge(**algoritmo_kwds)
    elif algoritmo=='rllasso':
      self.modelo=Lasso(**algoritmo_kwds)
    elif algoritmo=='lsvm':
      if 'kernel' not in algoritmo_kwds:
        algoritmo_kwds['kernel']='linear'
      elif algoritmo_kwds['kernel']!='linear':
        raise ValueError('Para o funcionamento correto da classe, utilize o kernel linear.')
      self.modelo=SVR(**algoritmo_kwds)
    elif algoritmo=='rarvore':
      self.modelo=DecisionTreeRegressor(**algoritmo_kwds)
    elif algoritmo=='rbfsvm':
      if 'kernel' not in algoritmo_kwds:
        algoritmo_kwds['kernel']='rbf'
      elif algoritmo_kwds['kernel']!='rbf':
        raise ValueError('Para o funcionamento correto da classe, utilize o kernel rbf.')
      self.modelo=SVR(**algoritmo_kwds)

    else:
      raise ValueError('Algoritmo Não Implementado')
    self.alg=algoritmo
    self.alg_kwds=algoritmo_kwds
    self.X=None
    self.y=None
    self.X_train=None
    self.y_train=None
    self.X_test=None
    self.y_test=None
    self.X_train_norm=None
    self.X_test_norm=None
    self.aval=None
    if normalizador=='minmax':
      self.norm=MinMaxScaler()
    elif normalizador=='standard':
      self.norm=StandardScaler()
    elif normalizador is None:
      self.norm=None
    else:
      raise ValueError('Normalizador Não Implementado')

  def __str__(self):
    if self.X_train is None:
      return f'Wrapper do Algoritmo {self.modelo} do Sklearn, Não Treinado'
    else:
      return f'Wrapper do Algoritmo {self.modelo} do Sklearn, Treinado'

  def __repr__(self):
    return f"RegDados1Atr(algoritmo='{self.alg}',algoritmo_kwds={self.alg_kwds})"

  def norm_treinar_aval(self,X,y,tts_rs=None):

    X_train,X_test,y_train,y_test=tts(X,y,random_state=tts_rs)
    if self.norm is not None:
      self.norm.fit(X_train)
      X_train_norm=self.norm.transform(X_train)
      X_test_norm=self.norm.transform(X_test)
      self.modelo.fit(X_train_norm,y_train)
      r2={'Treino':self.modelo.score(X_train_norm,y_train),'Teste':self.modelo.score(X_test_norm,y_test)}
      self.X_train_norm,self.X_test_norm=X_train_norm,X_test_norm
    else:
      self.modelo.fit(X_train,y_train)
      r2={'Treino':self.modelo.score(X_train,y_train),'Teste':self.modelo.score(X_test,y_test)}
    r2_df=pd.DataFrame(r2,index=['{}({}={})'.format(self.alg,list(self.alg_kwds.keys()),list(self.alg_kwds.values()))])
    self.X,self.y,self.X_train,self.y_train,self.X_test,self.y_test,self.aval=X,y,X_train,y_train,X_test,y_test,r2_df

  def dispersao_modelo(self,conjunto='treino',X_label='Atributo',y_label='Valor-Chave',salvar_grafico=False,
                       savefig_kwds={'dpi':150,'bbox_inches':'tight','fname':'grafico_fmodelo.jpg',
                                     'pad_inches':0.01},subplots_kwds={'figsize':(19,12)},xy_r2=None,modelo=None):
    if conjunto=='treino':
      if self.X_train_norm is not None:
        X=self.X_train_norm
      else:
        X=self.X_train
      y=self.y_train

    elif conjunto=='teste':
      if self.X_test_norm is not None:
        X=self.X_test_norm
      else:
        X=self.X_test
      y=self.y_test

    elif conjunto=='todos':
      if self.X_train_norm is not None and self.X_test_norm is not None:
        X=np.append(self.X_train_norm,self.X_test_norm,axis=0)
      else:
        X=self.X
      y=self.y

    seaborn.set_theme(context='notebook',style='darkgrid',palette='pastel',font='Monospace')
    fig,ax=mpl.pyplot.subplots(**subplots_kwds)
    ax.scatter(X,y,s=20,color='cyan')
    for xlabel in ax.get_xticklabels():
      xlabel.set_fontproperties({'size':16})
    for ylabel in ax.get_yticklabels():
      ylabel.set_fontproperties({'size':16})
    ax.set_title('Função do Modelo Sob o Gráfico de Dispersão (Atributo vs. Valor-Chave)',
                 fontdict={'size':20,'weight':'bold'})
    ax.set_xlabel(X_label,fontdict={'size':15,'weight':'bold'},labelpad=40)
    ax.set_ylabel(y_label,fontdict={'size':15,'weight':'bold'},labelpad=40)
    X_r=np.linspace(X.min()-0.1,X.max()+0.2,1000)
    if modelo is None:
      y_prev=self.modelo.predict(X_r.reshape(-1,1))
    else:
      y_prev=modelo.predict(X_r.reshape(-1,1))
    if self.alg=='lsvm':
      ax.plot(X_r,y_prev,color='red',linewidth=2.5,linestyle='dashed',
          label=f'$F(x)=({round(self.modelo.coef_[0][0],2)}\cdot x) + ({round(self.modelo.intercept_[0],2)})$')
    elif self.alg=='rarvore':
      ax.plot(X_r,y_prev,color='red',linewidth=2.5,linestyle='dashed',
          label=f'Previsões da Árvore de Decisão')
    elif self.alg=='rbfsvm':
      ax.plot(X_r,y_prev,color='red',linewidth=2.5,linestyle='dashed',
          label=f'Função da SVM com Kernel rbf')
    else:
      ax.plot(X_r,y_prev,color='red',linewidth=2.5,linestyle='dashed',
          label=f'$F(x)=({round(self.modelo.coef_[0],2)}\cdot x) + ({round(self.modelo.intercept_,2)})$')
    if xy_r2 is None:
      xy_r2=(ax.get_xlim()[0]+ax.get_xlim()[1]/20,ax.get_ylim()[1]-ax.get_ylim()[1]/10)
    ax.legend(loc='upper left',fontsize=20)
    ax.annotate(f'$R^2 = {round(self.modelo.score(X,y),2)}$',xy_r2,size=20)
    if salvar_grafico==True:
      mpl.pyplot.savefig(**savefig_kwds)
    mpl.pyplot.close('all')
    return (fig,ax)

  def teste_val_param(self,parametro,valores,X,y,tts_rs=None,plotar=False,
                     salvar_grafico=False,savefig_kwds={'dpi':150,'bbox_inches':'tight',
                                                        'fname':'grafico_fmodelo.jpg',
                                                        'pad_inches':0.01},subplots_kwds={'figsize':(19,12)}):

    norm=self.norm
    X_train,X_test,y_train,y_test=tts(X,y,random_state=tts_rs)

    modelos=[]
    for v in valores:
      if self.alg=='rlmq':
        modelos.append((v,LinearRegression(**{f'{parametro}':v})))
      elif self.alg=='rlridge':
        modelos.append((v,Ridge(**{f'{parametro}':v})))
      elif self.alg=='rllasso':
        modelos.append((v,Lasso(**{f'{parametro}':v})))
      elif self.alg=='lsvm':
        modelos.append((v,SVR(**{'kernel':'linear',f'{parametro}':v})))
      elif self.alg=='rarvore':
        modelos.append((v,DecisionTreeRegressor(**{f'{parametro}':v})))
      elif self.alg=='rbfsvm':
        modelos.append((v,SVR(**{'kernel':'rbf',f'{parametro}':v})))
      else:
        raise ValueError('Algoritmo Não Implementado')

    r2t=[]
    r2tt=[]
    ind=[]
    p_ord=[]
    for vm in modelos:
      if norm is not None:
        norm.fit(X_train)
        X_train_norm=norm.transform(X_train)
        X_test_norm=norm.transform(X_test)
        vm[1].fit(X_train_norm,y_train)
        r2t.append(vm[1].score(X_train_norm,y_train)),
        r2tt.append(vm[1].score(X_test_norm,y_test))
        ind.append(f'{self.alg}({parametro}={vm[0]})')
        p_ord.append(vm[0])
      else:
        vm[1].fit(X_train,y_train)
        r2t.append(vm[1].score(X_train,y_train))
        r2tt.append(vm[1].score(X_test,y_test))
        ind.append(f'{self.alg}({parametro}={vm[0]})')
        p_ord.append(vm[0])

    r2_df= pd.DataFrame({'Treino':r2t,'Teste':r2tt},index=ind)
    self.aval=pd.concat([self.aval,r2_df],join='outer')

    if plotar==True:
      seaborn.set_theme(context='notebook',style='darkgrid',palette='pastel',font='Monospace')
      fig,ax=mpl.pyplot.subplots(**subplots_kwds)
      ax.plot(p_ord,r2_df['Treino'],color='red',label='Treino',linewidth=2,linestyle='dashed')
      ax.plot(p_ord,r2_df['Teste'],color='green',label='Teste',linewidth=2)
      ax.set_xlabel(f'{parametro}',fontdict={'size':15,'weight':'bold'},labelpad=40)
      ax.set_ylabel(f'Avaliação',fontdict={'size':15,'weight':'bold'},labelpad=40)
      for xlabel in ax.get_xticklabels():
        xlabel.set_fontproperties({'size':16})
      for ylabel in ax.get_yticklabels():
        ylabel.set_fontproperties({'size':16})
      if salvar_grafico==True:
        mpl.pyplot.savefig(**savefig_kwds)
      mpl.pyplot.close('all')
      return (fig,ax)

    else:
      return modelos,r2_df

  def plotar_arvore(self,profundidade_max=None,no_rotulos=None,
                  precisao=2,pintar=True,fontdict=None,arrowdict=None,
                  fontdict_t=None,bg_color='white'):
    seaborn.reset_orig()

    if self.alg!='rarvore':
      raise ValueError('Algoritmo utilizado não é Árvore de Decisão')

    if self.X_train is None:
      raise ValueError('Árvore ainda não foi Treinada.')

    if fontdict is None:
      fontdict={'family':'Monospace','size':11,'weight':'bold'}
    elif type(fontdict)!=dict:
      raise ValueError('Propriedades da fonte tem de estar em formato de dicionário.')

    if no_rotulos is None:
      no_rotulos=['Atributo','Erro','#Exemplos','Valor']

    elif type(no_rotulos)!=list:
      raise ValueError('Rótulos dos nós tem de estar em formato de lista.')
    elif len(no_rotulos)!=4:
      raise ValueError('Lista de Rótulos precisa conter 4 item referentes a [Atr,Erro,Qtd Exemplos,Valor] respectivamente.')

    if arrowdict is None:
      arrowdict={'linewidth':2}
    elif type(arrowdict)!=dict:
      raise ValueError('Propriedades das Setas tem de estar em formato de dicionário.')

    if fontdict_t is None:
      fontdict_t={'family':'Monospace','size':20,'weight':'bold'}
    elif type(fontdict_t)!=dict:
      raise ValueError('Propriedades da fonte do título tem de estar em formato de dicionário.')

    fig,ax=mpl.pyplot.subplots(figsize=(24,12))
    mpl_txts=plot_tree(self.modelo,max_depth=profundidade_max,rounded=True,precision=precisao,filled=pintar)
    txt_labels=[]
    for i in mpl_txts:
      node_txt=i.get_text()
      labels=[i.split(' ')[0] for i in re.findall('((?=\s*)*.*(?=\s))',node_txt) if i.split(' ')[0]!='']
      txt_labels.append((node_txt,labels))
    rdict=[(k,v) for k,v in zip([i for i in txt_labels[0][1]],no_rotulos)]
    txt_r=[]
    for txt in [i[0] for i in txt_labels]:
      for (a,r) in rdict:
        txt=txt.replace(a,r)
      txt_r.append(txt)
    for mt,r in zip(mpl_txts,txt_r):
      mt.set(text=r,fontproperties=fontdict)
      mt.get_bbox_patch().set(linewidth=2)
    props=[i.arrow_patch.set(**arrowdict) for i in mpl_txts]
    fig.suptitle('Representação Gráfica da Árvore de Decisão',fontproperties=fontdict_t)
    fig.patch.set_facecolor(bg_color)
    mpl.pyplot.close('all')
    return (fig,ax)

  def validacao_cruzada(self,X,y,cv=5,ncpus=None):

    if self.alg=='rlmq':
      m=LinearRegression(**self.alg_kwds)
    elif self.alg=='rlridge':
      m=Ridge(**self.alg_kwds)
    elif self.alg=='rllasso':
      m=Lasso(**self.alg_kwds)
    elif self.alg=='lsvm':
      if 'kernel' not in self.alg_kwds:
        self.alg_kwds['kernel']='linear'
      elif self.alg_kwds['kernel']!='linear':
        raise ValueError('Para o funcionamento correto da classe, utilize o kernel linear.')
      m=SVR(**self.alg_kwds)
    elif self.alg=='rarvore':
      m=DecisionTreeRegressor(**self.alg_kwds)

    elif self.alg=='rbfsvm':
      if 'kernel' not in self.alg_kwds:
        self.alg_kwds['kernel']='rbf'
      elif self.alg_kwds['kernel']!='rbf':
        raise ValueError('Para o funcionamento correto da classe, utilize o kernel rbf.')
      m=SVR(**self.alg_kwds)

    else:
      raise ValueError('Algoritmo Não Implementado')
    cv_scores=cross_val_score(estimator=m,X=X,y=y,cv=KFold(n_splits=cv,shuffle=True),n_jobs=ncpus)
    media_cv=cv_scores.mean()
    return {'scores':cv_scores,'media':media_cv}

  def prever(self,valores):

    if self.X_train is None:
      raise ValueError('Modelo ainda não foi treinado.')

    if type(valores)==int or type(valores)==float:
      pred=np.array([float(valores)]).reshape(-1,1)
    elif type(valores)==list:
      pred=np.array(valores).reshape(-1,1)
    elif type(valores)==np.ndarray:
        pred=valores.reshape(-1,1)
    else:
      raise ValueError('Formato de valores passados não permitido')

    prev=self.modelo.predict(pred)
    return prev[0]

def carregar_pkl_treinado():
    X_ig=np.load('RegHelper/X_ig.npy')
    y_ig=np.load('RegHelper/y_ig.npy')
    arvore=Reg1Atr('rarvore')
    arvore.norm_treinar_aval(X_ig,y_ig,tts_rs=0)
    return arvore
def carregar_pkl():
    X_ig=np.load('RegHelper/X_ig.npy')
    y_ig=np.load('RegHelper/y_ig.npy')
    arvore=Reg1Atr('rarvore')
    return arvore

def carregar_dados():
  X_ig=np.load('RegHelper/X_ig.npy')
  y_ig=np.load('RegHelper/y_ig.npy')
  return (X_ig.ravel(),y_ig)
