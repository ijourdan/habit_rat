import numpy as np
import math
import matplotlib.pyplot as plt
import ratlibfunc
import torch

#%%
# %%
# get_ipython().run_line_magic('matplotlib', 'inline') #
# % matplotlib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%%

cantidad_estados = 6
cantidad_acciones = 5
act = ''  # acción tomada
Rme = 0  # acumulado de rwrd
Dme = 0  # acumulado de tiempo.
#%%
V = torch.empty((cantidad_estados, 2))
# Valor de los estados V(estadp, tiempo)
# filas corresponden a estados
# columna 0 tiempo mas reciente
Q = torch.empty((cantidad_estados, cantidad_acciones))
#Costo C(a(t),s(t-1),a(t-1))
Casa = torch.empty((cantidad_acciones, cantidad_estados, cantidad_acciones))
#Probabilidades para alpha
#P(a(t)/s(t-1),a(t-1))
Pasa_reg = torch.empty((300, cantidad_acciones, cantidad_estados))
# Pas_reg: P(a(t)/s(t)) .... no confundir con Psa_reg
Pas_reg = torch.empty((300, cantidad_acciones, cantidad_estados))

#%%
print(V)
print(Casa)


#%%

class CtxAsoc:
    """
    Esta corteza es la que de alguna manera aprende a resolver el problema.
    Vamos a plantear esta corteza por medio de un algorítmo de reinforcemente learning
    @author: ivan
    """

    def __init__(self):
        """
        Constructor.
        """

        # parametros
        self.cantidad_estados = 6
        self.cantidad_acciones = 5
        self.act = ''  # acción tomada
        self.Rme = 0  # acumulado de rwrd
        self.Dme = 0  # acumulado de tiempo.

