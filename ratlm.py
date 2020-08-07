#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:06:04 2018
@author: ivan
"""
##
import numpy as np
import math
# import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import ratlibfunc
##
# %%
# get_ipython().run_line_magic('matplotlib', 'inline') #
# % matplotlib




class Rat:
    """

    @author: ivan
    """
    def __init__(self):
        """
        Constructor.
        """

        #Variables de estado y accion
        self.set_action = ['PL', 'EM']  # PL:palanca EM:entra magazine
        # descripción del estado 'R' es reward, 'D' es delay temporal
        self.estado = pd.DataFrame({'R': [1, 0, 0], 'D': [1, 1, 1]})
        # transicion estado frente a acción tomada
        self.trans = pd.DataFrame({'PL': [1, 1, 1], 'EM': [2, 0, 2]})

        self.estN = 0  # nro de estado
        self.estN1 = 0  # siguiente estado
        self.act = ''  # accion tomada
        self.Rme = 0  # Acumulado de reward
        self.Dme = 0  # Acumulado de tiempo

        # PARAMETROS DE APRENDIZAJE
        self.V = np.zeros((3, 2))
        # Valor de los estados
        # filas corresponden a estados
        # columna 0 tiempo mas reciente
        self.Q = pd.DataFrame(index=[0, 1, 2], columns=self.set_action, dtype='float')

        #Costo C(a(t),s(t-1),a(t-1))
        self.Casa = pd.DataFrame(index=self.set_action, columns=pd.MultiIndex.from_product([[0, 1, 2], self.set_action]),
                                 dtype='float')
        #Probabilidades para alpha
        #P(a(t)/s(t-1),a(t-1))
        self.Pasa_reg = pd.DataFrame(index=list(np.arange(300)),
                                 columns=pd.MultiIndex.from_product([self.set_action,[0,1,2], self.set_action]),
                                 dtype='float')

        #Pas_reg: P(a(t)/s(t)) .... no confundir con Psa_reg
        self.Pas_reg = pd.DataFrame(index=list(np.arange(300)),
                                 columns = pd.MultiIndex.from_product([self.set_action,[0,1,2]]),
                                 dtype='float')

        # Registro de estado y acción anterior (n-1).
        self.as_previo={'act':'', 'est':0}
        #iniciamos en forma aleatoria.
        if np.random.rand()>0.5:
            self.as_previo['act'] = self.set_action[0]
        else:
            self.as_previo['act'] = self.set_action[1]

        # Reward
        self.R_reg = np.zeros((3, 10))  # registro acumulativo
        self.R = np.zeros((3, 1))  # estimacion del rwrd medio por estado
        # Demora
        self.D_reg = np.zeros((3, 10))  # registro acumulativo
        self.D = np.zeros((3, 1))  # estimación del delay medio por estado
        self.step = 0  # cantidad de pasos en el juego
        self.eta = 0.001
        self.nivel = 0.25
        self.Psa = [] #registro de las probabilidades condicionales P(a/s) a los fines de graficación

        # inicializamos matrices
        self.Q[:] = np.zeros(shape=self.Q.shape)
        self.V_calc()
        self.step = 0
        self.Psa = []
        self.Casa[:] = np.zeros(shape=self.Casa.shape)
        self.Pasa_reg[:] = np.zeros(shape=self.Pasa_reg.shape)
        self.Pas_reg[:] = np.zeros(shape=self.Pas_reg.shape)

        # PARAMETROS PARA DETERMINAR EL ERROR DE PREDICCIÓN
        dim=6 # 3 estados y dos acciones por estado
        self.W = np.zeros((dim,1))  # pesos que se aprenden.
        self.alpha_eta=0.1 #coerficiente para definir a eta
        aux=ratlibfunc.base_plus(dim)
        col2=[[0,1,2],self.set_action]
        col2 = pd.MultiIndex.from_product(col2)
        self.mapping_states={}
        for j in range(len(col2)):
            self.mapping_states[col2[j]]=aux[j]
        #Se determino con self.mapping_states un mapeo ortogonal para los estado


#FUNCIONES


#===================Funcion de reseteo de aprendizaje

    def start(self):
        # más o menos lo mismo que el constructor.
        # permite reiniciar el entrenamiento
        # estado
        self.estN = 0  # anterior
        self.estN1 = 0  # actual
        # acciones
        self.act = ''  # anterior


        self.Rme = 0  # Acumulado de reward
        self.Dme = 0  # Acumulado de tiempo

        # inicializamos matrices
        #self.Pscsa[:] = np.zeros(shape=self.Pscsa.shape)
        self.Q[:] = np.zeros(shape=self.Q.shape)
        self.V_calc()
        self.step = 0
       # self.Psa_reg[:] = np.zeros(shape=self.Psa_reg.shape)
        self.Psa = []
        dim = 6
        self.W = np.zeros((dim,1))  # pesos que se aprenden.

#==============Funciones de aprendizaje

    def V_calc(self):
        self.V[:, 1] = self.V[:, 0]
        for t in [0, 1, 2]:
            self.V[t][0] = self.Q.loc[t].max()

    def eta_Q(self):
        estimadorerror=self.estado.loc[self.estN1]['R'] - ratlibfunc.valuenn(self.W, self.mapping_states[(self.estN,self.act)])
        value=ratlibfunc.sigmoid(10*(estimadorerror-0.3))
        return 1*value+0.001

    def Q_calc(self):
        # Q vamos a implementar Asynchronous Value Iteration para determinar
        # el valor de Q.

        s = self.estN
        a = self.act
        s1 =self.estN1
        aux = (self.R[s1] - self.D[s1] * (self.Rme / self.Dme) + self.V[s1, 0])
        if np.linalg.norm(aux) != 0 :
            etav = self.eta_Q()
            self.Q.loc[s][a] = (1-etav) * self.Q.loc[s][a] + etav * aux
            #self.Q.loc[s][a] += self.eta_Q() * aux


    #funciones correpondientes a habitos


    def deltan(self):
        """
        Funcion que es parte del circuito de formación de habitos:
        deltan(self) calcula la señal de error de diferencia temporal (TD), que es parte de la función de Costo
        C(s,a,a') necesaria para definir las macros de acción {a,a1}.
        :return:(rwrd - delay * (self.Rme / self.Dme) + self.V[s1, 0])-self.V[s, 1]
        """
        s = self.estN
        s1 = self.estN1
        rwrd = self.estado.loc[s]['R']
        delay = self.estado.loc[s]['R']
        return (rwrd - delay * (self.Rme / self.Dme) + self.V[s1, 0])-self.V[s, 1]


    def Pasa(self,at,at1,st1):
        """
        Determina la Probabilidad P(a(t)/a(t-1),s(t-1)) necesaria para el calculo del paramete alpha(t)
        :param at: acción tomada en el estado estN y que implica llegar a estN1
        :param at1: acción tomada en es estado st1, y que como consecuencia se pasó a estN,
        :param st1: estado previo a estN del cual se pasó a estN
        :return: float  P(a(t)/a(t-1),s(t-1))
        """
        aux=list(self.Pasa_reg.loc[:][at,at1,st1])
        auxcantidad = 0
        for i in self.set_action:
            auxcantidad +=  sum(list(self.Pasa_reg.loc[:][i,at1,st1]))
        if auxcantidad == 0:
            return 0
        else:
            return float(sum(aux))/auxcantidad

    def Pas(self,at,st):
        """
        Determina la Probabilidad P(a(t)/s(t)) necesaria para el calculo del paramete alpha(t)
        :param at: acción tomada en el estado estN y que implica llegar a estN1
        :param st: estado estN
        :return: float  P(a(t)/s(t))
        """
        aux=list(self.Pas_reg.loc[:][at,st])
        auxcantidad = 0
        for i in self.set_action:
            auxcantidad += sum(list(self.Pas_reg.loc[:][i, st]))
        if auxcantidad == 0:
            return 0
        else:
            return float(sum(aux)) / auxcantidad

    def Casa_lear(self):
        etav=self.eta_Q()
        for at in self.set_action:
            for st in [0,1,2]:
                auxpas = self.Pas(at, st)
                for at1 in self.set_action:
                    if auxpas == 0:
                        alfa = 0
                    else:
                        auxpasa=self.Pasa(at,st,at1)
                        alfa = auxpasa/auxpas
                    self.Casa.loc[at][st,at1]=(1-etav)*self.Casa.loc[at][st,at1]+etav*alfa*self.deltan()

    def learn(self):
        #aprendizaje del modelo de RL
        self.Q_calc()
        self.V_calc()
        # aprendizaje de la neurona que estima el reward (adapta self.W)
        x = self.mapping_states[(self.estN, self.act)]
        (out, self.W) = ratlibfunc.valuenn(self.W, x, learn=True, r=self.estado.loc[self.estN1]['R'],
                                           alpha_eta=self.alpha_eta)
        self.Casa_lear()

        # Finalmente, actualizamos el estado
        self.estN = self.estN1

#=========== Funciones de toma de decision

    def think(self):  # the policy o mas o menos
        self.nivel = 0.2 #hay un 20% de iniciativa exploratoria. Aqui puede entrar efectos de ACH
        #self.nivel = 0.75 * np.exp(-0.1 * self.step) + .2  # va pasando de algo muy exploratorio a no tanto
        if float(np.random.rand(1)) > self.nivel:  # vamos a elegir e-gready
            if self.Q.loc[self.estN][self.set_action[0]] == self.Q.max(axis=1)[self.estN]:
                self.act = self.set_action[0]
            else:
                self.act = self.set_action[1]
        else:  # elegimos al azar (exploratorio)
            if float(np.random.rand(1)) > .5:
                self.act = self.set_action[0]
            else:
                self.act = self.set_action[1]


#============== Funciones de ejecucion

    def game(self):
        self.step += 1
        self.estN1 = self.trans.loc[self.estN][self.act]
        self.Rme += self.estado.loc[self.estN1]['R']
        self.Dme += self.estado.loc[self.estN1]['D']
        # Calculo de R
        self.R_reg[self.estN1, :] = np.roll(self.R_reg[self.estN1, :], 1)
        self.R_reg[self.estN1, 0] = self.estado.loc[self.estN1]['R']
        self.R = self.R_reg.mean(axis=1)
        self.R = self.R.reshape(3, 1)
        # Calculo de D
        self.D_reg[self.estN1, :] = np.roll(self.D_reg[self.estN1, :], 1)
        self.D_reg[self.estN1, 0] = self.estado.loc[self.estN1]['D']
        self.D = self.D_reg.mean(axis=1)
        self.D = self.D.reshape(3, 1)

        aux = pd.DataFrame(list(self.Pas_reg.loc[0]), index=pd.MultiIndex.from_product([self.set_action,[0,1,2]]),
                                 dtype='float')
        for st in [0, 1, 2]:
            for at in self.set_action:
                aux.loc[at,st]=self.Pas(at,st)

        self.Psa.extend(list(aux[0]))

        #Estadisticos requeridos para HABITOS
        # Asignacion de Pasa_reg y Pas_reg
        #Pasa_reg:
        self.Pasa_reg = self.Pasa_reg.shift(1)
        self.Pasa_reg.loc[0] = 0
        self.Pasa_reg.loc[0][self.act,self.as_previo['est'],self.as_previo['act']]=1
        #Pas_reg:
        self.Pas_reg = self.Pas_reg.shift(1)
        self.Pas_reg.loc[0] = 0
        self.Pas_reg.loc[0][self.act,self.estN]=1
        #actualizamos self.as_previo
        self.as_previo['est']=self.estN
        self.as_previo['act']=self.act


    def play(self, NumPlay=100):
        for i in range(NumPlay):
            self.think()
            self.game()
            self.learn()


#===========Funciones para observar a Rata

    def desc(self):
        print('estN=' + str(self.estN))
        print('accion=' + str(self.act))
        print('Q=' + str(self.Q))
        print('V=' + str(self.V))


    def plotPacs(self, acc, est):
        """ Grafica P(acc/est)  """
        aux = np.asarray(self.Psa)
        aux = aux.reshape(int(len(aux) / 6), 6)
        dfPas = pd.DataFrame(aux, columns=pd.MultiIndex.from_product([self.set_action,[0, 1, 2]]))
        aux = dfPas[acc,est] # / dfPsa[est].sum(axis=1)
        aux.plot()
        #return dfPsa

    def observaW(self):
        for j in list(self.mapping_states.keys()):
            print(str(j)+' : '+str(ratlibfunc.valuenn(self.W,self.mapping_states[j])))




class Mouse:
    def __init__(self, NN, MM):
        # parametros generales
        self.NN = NN  # Cantidad de Filas
        self.MM = MM  # Cantidad de Columnas
        self.END1 = 1  # Goal 1
        self.END2 = self.MM  # Goal 2
        self.START = (self.NN - 1) * self.MM + (self.MM + 1) / 2  # Estado (posición) en inicio cuando se crea (n=0)
        self.AA = 4  # Cantidad de Acciones =4
        # a=0 avanza
        # a=1 retrocede
        # a=2 derecha
        # a=3 izquierda

        # parametros de estado
        self.stat = self.START  # Estado (posición) en inicio cuando se crea (n=0)
        self.stat1 = 0  # auxiliar para resguardo de Estado
        self.forbidden_states = np.array([-100])  # Estados a los que no se puede pasar (pared).
        # No pueden estar ni inicio ni 1 ni MM.
        # Se inicializa en -100 ya que nunca puede ser un estado

        # parametros de accion
        self.action = 0  # Accion seleccionada/ejecutada (n)
        self.action1 = 0  # Accion para resguardo de accion.

        # parametros de recompenza
        self.rwrd = 0  # Recompenza
        self.rwrd_map = np.zeros(self.NN * self.MM)  # Mapa de recompenza. Inicia en cero

        # parametros de aprendizaje
        self.disc = 0.9  # discount
        self.alpha = 0.1  # learning rate
        self.lam = 0.9  # trace decay

        # Valor Q(s,a)
        self.Qsa = np.random.rand(self.MM * self.NN, self.AA)  # Vector de valor Q(s,a)
        self.eQsa = np.zeros((self.MM * self.NN, self.AA), dtype=float)  # Vector error de valor eQsa

        # Gráficas para mostrar la evolución del aprendizaje.
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def plt_game(self):  # grafica el juego actual en estado inicial.
        rect = plt.Rectangle((0, 0), 1, 1, color='w')  # limpia la grafica
        self.ax.add_patch(rect)
        # lugar de rwrd 1
        auxlocation = (0, (self.NN - 1) / self.NN)
        rect = plt.Rectangle(auxlocation, 1 / self.MM, 1 / self.NN, color='y')
        self.ax.add_patch(rect)
        # lugar de rwrd 2
        aux1 = (self.MM - 1) // self.MM  # parte entera
        aux2 = (self.MM - 1) % self.MM  # resto
        auxlocation = (aux2 / self.MM, (self.NN - 1 - aux1) / self.NN)
        rect = plt.Rectangle(auxlocation, 1 / self.MM, 1 / self.NN, color='y')
        self.ax.add_patch(rect)

        # estado inicial
        aux1 = (self.stat - 1) // self.MM  # parte entera
        aux2 = (self.stat - 1) % self.MM  # resto
        auxlocation = (aux2 / self.MM, (self.NN - 1 - aux1) / self.NN)
        rect = plt.Rectangle(auxlocation, 1 / self.MM, 1 / self.NN, color='g')
        self.ax.add_patch(rect)

        # si hay un laberinto definido
        for auxts in self.forbidden_states:
            if auxts > 0:
                aux1 = (auxts - 1) // self.MM  # parte entera
                aux2 = (auxts - 1) % self.MM  # resto
                auxlocation = (aux2 / self.MM, (self.NN - 1 - aux1) / self.NN)
                rect = plt.Rectangle(auxlocation, 1 / self.MM, 1 / self.NN, color='k')
                self.ax.add_patch(rect)

    def pltstat(self):  # Grafica el estado y otra info.
        aux1 = (self.stat1 - 1) // self.MM  # parte entera
        aux2 = (self.stat1 - 1) % self.MM  # resto
        auxlocation = (aux2 / self.MM, (self.NN - 1 - aux1) / self.NN)
        rect = plt.Rectangle(auxlocation, 1 / self.MM, 1 / self.NN, color='#9159f9')
        self.ax.add_patch(rect)

        aux1 = (self.stat - 1) // self.MM  # parte entera
        aux2 = (self.stat - 1) % self.MM  # resto
        auxlocation = (aux2 / self.MM, (self.NN - 1 - aux1) / self.NN)
        rect = plt.Rectangle(auxlocation, 1 / self.MM, 1 / self.NN, color='g')
        self.ax.add_patch(rect)
        plt.pause(0.05)

    # definiciones de laberinto ==============

    def labyrinth_open(self):
        self.forbidden_states = np.array([-100])

    def labyrinth_T(self):
        data = []

        for k in range(1, self.NN):
            data.append(k * self.MM + np.floor(self.MM / 2))  # estados a la izq del camino vertical
            data.append(k * self.MM + np.floor(self.MM / 2) + 2)  # estados a la der
        aux = self.MM + np.floor(self.MM / 2)
        aux = aux.astype(np.int64)
        for k in range(self.MM + 1, aux):
            data.append(k)  # estados sup del camino horiz (izq)
            data.append(k + np.floor(self.MM / 2) + 2)  # estados sup del camino horiz (der)

        self.forbidden_states = np.reshape(data, len(data))  # aqui reservo la cantidad de estados

    # def labyrinth_Y(self):

    # def labyrinth_random(self,porcentual):
    # ==============================================

    def Inicio(self):  # Inicia el experimento
        self.stat = self.START  # se lleva a estado n=0
        self.action = self.policy()  # accion
        # self.rwrd_map
        self.plt_game()

    # funciones de ejecución de acciones

    def avanza(self):
        if (self.stat - self.MM) > 0:  # OK
            if not ((self.forbidden_states == (self.stat - self.MM)).max()):  # true => estado permitido
                self.stat = self.stat - self.MM

    def retrocede(self):
        if (self.stat + self.MM) < (self.NN * self.MM + 1):  # OK
            if not ((self.forbidden_states == (self.stat + self.MM)).max()):  # true => estado permitido
                self.stat = self.stat + self.MM

    def izq(self):
        auxstmin = np.floor(self.stat / self.MM) * self.MM + 1
        if (self.stat - 1) >= auxstmin:  # OK
            if not ((self.forbidden_states == (self.stat - 1)).max()):  # true => estado permitido
                self.stat = self.stat - 1

    def der(self):
        auxstmax = np.ceil(self.stat / self.MM) * self.MM
        if (self.stat + 1) <= auxstmax:  # OK
            if not ((self.forbidden_states == (self.stat + 1)).max()):  # true => estado permitido
                self.stat = self.stat + 1

    # funciones de ejecucion de tarea

    # policy (action selection from Q(s,a))
    def policy(self):  # pi(s,a) determina la acción "a" a tomar de acuerdo al estado "s"
        # typepol =0 => greedy, =1 => exploratoria.
        # if typepol==1: #exploratoria, a implementar. ahora es greedy
        return np.asscalar(self.Qsa[int(self.stat) - 1, :].argmax())

    # funciones de recompenza

    def reward(self):
        if self.stat == 1:
            return 1
        elif self.stat == 5:
            return 1
        else:
            return 0

    # Ejecución de accion, toma de decision y aprendizaje
    def workmaze(self):  # esta funcion puede integrarse con los movimientos.
        if self.action == 0:
            self.avanza()
        elif self.action == 1:
            self.retrocede()
        elif self.action == 2:
            self.der()
        elif self.action == 3:
            self.izq()

    def working(self):
        oldQsa = self.Qsa.copy()  # resguardamos por si aparecen nan
        oldeQsa = self.eQsa.copy()  # resguardamos por si aparecen nan
        self.stat1 = self.stat  # resguardamos estado
        self.workmaze()  # se mueve en el laberinto. Se modifica self.stat.
        self.action1 = self.action  # resguardamos la accion
        self.action = self.policy()  # determinamos la futura accion a tomar de acuerdo al nuevo estado.
        # se modifica self.action.

        # a partir de aqui empieza a aprender.

        if self.stat1 == self.stat:  # es decir, con la accion no pudo moverse
            self.Qsa[int(self.stat) - 1, int(self.action1)] = -1e10  # no se toma mas esta accion cuando
            # se esta en el estado stat1=stat
            self.eQsa[int(self.stat) - 1, int(self.action1)] = 0
        else:
            aaux = np.asscalar(self.Qsa[int(self.stat) - 1, :].argmax())  # se busca el greedy, la acción con mas Q
            deltaQ = self.reward() + self.disc * self.Qsa[int(self.stat) - 1, aaux] - self.Qsa[
                int(self.stat1) - 1, self.action1]
            self.eQsa[int(self.stat1) - 1, int(self.action1)] = self.eQsa[int(self.stat1) - 1, int(self.action1)] + 1
            for auxs in range(0, self.NN * self.MM):
                for auxa in range(0, self.AA):
                    self.Qsa[auxs, auxa] = self.Qsa[auxs, auxa] + self.alpha * deltaQ * self.eQsa[auxs, auxa]
                    if self.action == aaux:  # si la accion futura es greedy se mantiene la traza
                        self.eQsa[auxs, auxa] = self.lam * deltaQ * self.eQsa[auxs, auxa]
                    else:
                        self.eQsa[auxs, auxa] = 0

        # chequeo de NaN
        if (math.isnan(self.Qsa.max()) ^ math.isnan(
                self.Qsa.max())):  # si hay nan, se tira este trial y se recuperan los datos
            self.Qsa = oldQsa.copy()
            self.eQsa = oldeQsa.copy()

    def checkgoal(self):
        if (self.stat == self.END1) ^ (self.stat == self.END2):  # llego a algun goal
            self.Inicio()
            return True
        else:
            return False

    def execute(self, arg1, arg2):
        kk = 0
        while not (self.checkgoal()) and (kk < arg1):
            self.working()
            if arg2 == 1:
                self.pltstat()
            kk += 1
        # print(kk)
        return kk
