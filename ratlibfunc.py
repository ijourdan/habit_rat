import numpy as np
#import math
#import pandas as pd


# Una coleccion de activation functions

def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1 - (x ** 2)
    return np.tanh(x)


def relu(x, derivative=False):
    if derivative:
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x


def arctan(x, derivative=False):
    if derivative:
        return np.cos(x) ** 2
    return np.arctan(x)


def step(x, derivative=False):
    if derivative:
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                x[i][k] = 1
            else:
                x[i][k] = 0
    return x


def squash(x, derivative=False):
    if derivative:
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = (x[i][k]) / (1 + x[i][k])
                else:
                    x[i][k] = (x[i][k]) / (1 - x[i][k])
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            x[i][k] = (x[i][k]) / (1 + abs(x[i][k]))
    return x


def gaussian(x, derivative=False):
    if derivative:
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                x[i][k] = -2 * x[i][k] * np.exp(-x[i][k] ** 2)
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            x[i][k] = np.exp(-x[i][k] ** 2)
    return x

# Generador de bases ortogonales
def base_plus(dimension):
    base=[]
    aux=np.random.rand(dimension).reshape(dimension,1)
    aux=aux/np.linalg.norm(aux)
    base.extend([aux])
    for i in range(dimension-1):
        aux = np.random.rand(dimension).reshape(dimension, 1)
        aux = aux / np.linalg.norm(aux)
        aux2 = [(base[j].transpose()).dot(aux) * base[j] for j in range(len(base))]
        aux=aux-np.sum(aux2,axis=0)
        aux=aux/np.linalg.norm(aux)
        base.extend([aux])
    return base



#neurona

"""
La organización de las funciones de la neurona que estima el valor de recompenza es la siguiente:
valuenn :
        DeltaW:
                weight_for_x: --> func_eta:
"""

def weight_for_x(x):
    """
    w = weight_for_x(x)
    Determina un vector de peso tal que minimice el error:
    E=||X-W.dot(X)||

    Si r=o, entonces w=0
    """
    eigval = np.linalg.eigvals(x.dot(x.transpose()))
    flambda = 1 / (eigval.max() * 2)
    mid = np.eye(len(x))
    auxa = flambda * np.linalg.inv(mid - x.dot(x.transpose()) * flambda)
    w = auxa.dot(x)
    return w

def func_eta(r, w_actual, x, alpha_eta=0.1):
    """
    Calcula el parametro proporcional para la actualización del peso w:
    w(n+1) = w(n) + eta w_x
    :param r: reward para el estado x
    :param w_actual: peso a ser actualizado
    :param x: vector de estado
    :param alpha_eta: Parámetro de modulacion de velocidad de aprendizaje (default 0.1)
    :return: un numero
    """
    eta_num = alpha_eta * (r - (w_actual.transpose()).dot(x))
    return eta_num


def DeltaW(r,w_actual,x,alpha_eta=0.1):
    """
    DeltaW(r,w_actual,thr,x,alpha_eta=0.1):
    Determina la variación del vector de peso de la red: w(n+1)=w(n)+DeltaW
    :param r: reward para el estado x
    :param w_actual: peso a ser actualizado
    :param x: vector representativo del estado
    :param alpha_eta: Parámetro de modulacion de velocidad de aprendizaje (default 0.1)
    :return: DeltaW retorna un nparray
    """
    w = weight_for_x(x)
    eta_value=func_eta(r,w_actual,x,alpha_eta)
    return eta_value*w

def valuenn(w_actual,x,learn=False,r=None,alpha_eta=0.1):
    """
    valuenn(w_actual,x,learn=False,r=None,thr=None,alpha_eta=0.1)
    Entrena la red neuronal.
    :param w_actual: peso a ser actualizado
    :param x: vector representativo del estado
    :param learn: False (default). True entonces modifica los pesos.
    :param r:reward para el estado x
    :param alpha_eta: Parámetro de modulacion de velocidad de aprendizaje (default 0.1)
    :return: Si learn=False, return=W.dot(X) Si learn=True, return=(W.dot(X), W)
    """
    if learn:
        if None == r:
            print('Error: learn=True requiere definir r y thr')
            raise NameError('FaltaParam')
        else:
            w_actual = w_actual + DeltaW(r,w_actual,x,alpha_eta) #esto no cambia a w_actual original
        return float(((w_actual.transpose()).dot(x)).real) , w_actual
    else:
        return float(((w_actual.transpose()).dot(x)).real)









