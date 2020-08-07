#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:57:28 2018

@author: ivan
"""
##
import numpy as np
#import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import ratlm
import ratlibfunc
# In[dsd]:

# %%
##
set_action = ['PL','EM'] #PL:palanca EM:entra magazine
estado = pd.DataFrame({'R':[1,0,0],'D':[1,1,1]})
       #transicion estado frente a acción tomada
trans=pd.DataFrame({'PL':[1,1,1],'EM':[2,0,2]})

estN=0 #nro de estado
estN1=0 #siguiente estado
act='' #accion tomada
act1='' #accion a tomar
Rme=0 #Acumulado de reward
Dme=0 #Acumulado de tiempo
V=np.zeros((3,2)) 
        #Valor de los estados
        #filas corresponden a estados
        #columna 0 tiempo mas reciente
Q=pd.DataFrame(index=[0,1,2],columns=set_action)
        #Probabilidad de transcición de eStado Condicionada al eStado y la Acción
Pscsa=pd.DataFrame(index=[0,1,2,3],columns=pd.MultiIndex.from_product([[0,1,2],set_action]),dtype=float)
        # el 3er indicice es para ir contando la cantidad de veces que se expresó 
        # la condición.Es para calcular la probabilidad
step=0 #cantidad de pasos en el juego

R=pd.DataFrame(index=[0,1,2],columns=pd.MultiIndex.from_product([[0,1,2],set_action]),dtype='float')
        #Demora
D=pd.DataFrame(index=[0,1,2],columns=pd.MultiIndex.from_product([[0,1,2],set_action]),dtype='float')
        

Pscsa[:]=np.zeros(shape=Pscsa.shape,dtype='float')
Q[:]=np.random.rand(Q.shape[0],Q.shape[1])
R[:]=np.zeros(shape=R.shape)
D[:]=np.zeros(shape=D.shape)

Psa_reg=pd.DataFrame(index=list(np.arange(300)),columns=pd.MultiIndex.from_product([[0,1,2],set_action]),dtype='int')
Psa=[]
##
# %%
estN=0 
estN1=1 
act='PL'

# %% game

step+=1
estN1=trans.loc[estN][act]
Rme+=estado.loc[estN1]['R']
Dme+=estado.loc[estN1]['D']
R.loc[estN1][estN][act]+=estado.loc[estN1]['R']
D.loc[estN1][estN][act]+=estado.loc[estN1]['D']
# %% Pscsa update

auxNum=Pscsa.loc[3][estN,act] 
auxNum=1 if auxNum==0 else auxNum 
Pscsa.loc[3][estN,act]+=1
for i in [0,1,2]:
    if i==estN1:
        Pscsa.loc[i][estN,act]=float(Pscsa.loc[i][estN,act]*auxNum+1)/Pscsa.loc[3][estN,act]
    else:
        Pscsa.loc[i][estN,act]=float(Pscsa.loc[i][estN,act])*auxNum/Pscsa.loc[3][estN,act]
        

# %% Q update
        
for s in [0,1,2]:
    for a in set_action:
        aux=0
        for s1 in [0,1,2]:
            aux+=Pscsa.loc[s1][s][a]*(R.loc[s1][s][a]+R.loc[s1][s][a]*(Rme/Dme)+V[s1][0])
        
        Q.loc[s,a]=aux       
##
# %%
        
eta=0.1
action_map={a.set_action[i]:i for i in range(len(a.set_action))}
X=np.asarray([a.estN, action_map[a.act], 1 ])
X=X.reshape(len(X),1)
eigval=np.linalg.eigvals(X.dot(X.transpose()))
flambda=1/(eigval.max()*2)
Mid=np.eye(len(X))
A=eta*a.R[a.estN1]*np.linalg.inv(Mid/flambda-X.dot(X.transpose()))
W=A.dot(X)
W.transpose().dot(X)

##
# Algo de basura que lo vamos a necesitar en otro lado.
# mapeamos la entrada [acc , estado]
    action_map = {self.set_action[i]: i for i in range(len(self.set_action))}
    Y = np.asarray([self.estN, action_map[self.act]])
    Y = Y.reshape(len(Y), 1)
    # decorrelacionamos
    x = self.Uw.dot(Y) + self.Ub
    x = sigmoid(x)

##
def tsts(x):
    print(x)
    x=x+1
    print(x)
    return True

##
x=np.asarray(list(range(0,100)))
x.reshape(100,1)
x=x/100

y= ratlibfunc.sigmoid(5*(x-0.35))

plt.plot(x,y)
##
# para probar ratlm
# Creacion de rata

a=ratlm.Rat()
##
#  experimento 1: se entrena 2000 con rwrd en estado 2,
for i in range (0,2):
    if i==0:
        a.play(1000)
    else:
        a.play(3000)
    #cambia de lugar la recompenza (de estado 0 a estado 2)
    a.estado.loc[0]['R']=0
    a.estado.loc[2]['R']=1
    # otras 600 tirals más
    if i==0:
        a.play(1000)
    else:
        a.play(1000)
    #volvemos al original (rwrd en estado 0)
    a.estado.loc[0]['R']=1
    a.estado.loc[2]['R']=0
    # Graficamos la proba de que elija EM estando en 2.
##
a.plotPacs('PL',0)
a.plotPacs('EM',0)
a.plotPacs('EM',2)

##
# Experimento donde hay


aux = pd.DataFrame(list(a.Pas_reg.loc[0]), index=pd.MultiIndex.from_product([a.set_action,[0,1,2]]),dtype='float')
for st in [0, 1, 2]:
    for at in a.set_action:
        aux.loc[at,st]=a.Pas(at,st)

##
def tst(value,x):
     result = {
        'a': lambda x: x * 5,
        'b': lambda : print('test'),
        'c': lambda x: x+1
    }[value](x)
     return result

