"""
Librería con funciones de interacción con el ambiente.
Emite las señales sensoriales, ejecuta las modificaciones de ambiente cada vez que se realiza una ejecución
mediante una acción.

Procedimiento:
1.- Importar libreria:
> import ratlibenviro

2.- Crear el ambiente experimental:
> enviro = ratlibenviro.CajaSk(N_filas, N_columnas) #con N_filas y N_columnas nros impares
o tambien
> enviro = ratlibenviro.CajaSk()
donde queda definido la salida del sensor (enviro.sensor) como una matriz de 11 filas y 21 columnas

3.- Configuración de conducta a realizar.
enviro.schedule = 'RI'  # o también  'CR', 'RR', 'FI', 'RI'
enviro.schedule_param = 30  # esta definicion implica RI-30
enviro.rwrd_izq = True (False) # si debiera ir a la IZQ (DER) para luego PAL y recibir reward.

4.- Inicializa la simulación
Es algo así como poner al sujeto en la caja Skinner. Se da identidad al entorno de simulación.
> enviro.tarea_rl_init(self)

5.- Ejecución
Se ejecutan las acciones:
> enviro.tarea_rl(accion)
donde accion puede ser 'DER', 'IZQ', 'AVA', 'RET', 'NAM' o 'PAL'

no se obtiene un resultado, pero se modifican:
enviro.estado_caja
enviro.rwrd_out   # Reward que ofrece el nuevo estado, de acuerdo a la acción tomada.
enviro.delay_out   # Delay que se tiene en el nuevo estado, de acuero a la acción tomada
elf.trial_end   # Es True si se termino el trial, y pasa a False cuando se realiza un nuevo trial. En el caso de la
                simulación, corresponde cuando por medio de 'RET' se pasa al llamado estado 0 o inicial.

"""

import numpy as np
import pandas as pd
from cv2 import waitKey, imshow
import time


class CajaSk:
    def __init__(self, nn=11, mm=21):
        """
        Constructor
        """
        self.NN = nn  # Cantidad de Filas
        self.MM = mm  # Canitdad de Columnas

        self.sensor = np.zeros(shape=(self.NN, self.MM))
        self.set_acciones = ['DER', 'IZQ', 'AVA', 'RET', 'NAM', 'PAL']  # Son las acciones posibles
        self.set_schedules = ['CR', 'RR', 'FI',
                              'RI']  # CR Cont reinf, RR random reinf, FI Fixed interval, RI, Random Int
        self.interno_estados = []  # iniciarlo de acuerdo al experimento que se quiera generar.
        self.estado_caja = 0  # estado en el que se encuentra la caja
        self.transicion = pd.DataFrame([])  # matriz de trancision. Hay que inicializarla. Lo mejor es un Dataframe
        self.reward = pd.DataFrame([])  # matriz de recompensa. Hay que inicializarla. Lo mejor es un Dataframe
        self.delay = pd.DataFrame([])  # matriz de delay. Hay que inicializarla. Lo mejor es un Dataframe
        self.cond_0 = False  # condiciones para usar, por ejemplo cuando hay reward
        self.cond_1 = False  # otra condicion,

        # Configuración conducta
        self.schedule = 'CR'  # schedule de entrenamiento
        self.schedule_param = 1  # parametro del schedule
        self.rwrd_izq = False  # parametro de que lado define reward True a Izquiers
        # variables auxiliares para la conducta
        self.schedule_umbral = 0  # umbral que determina el cumplimiento de la condición.
        self.schedule_acumulador = 0  # acumulador de respuests
        # resultados de cada accion
        self.rwrd_out = 0  # Reward que ofrece el nuevo estado, de acuerdo a la acción tomada.
        self.delay_out = 0  # Delay que se tiene en el nuevo estado, de acuero a la acción tomada
        self.trial_end = False  # Indica si termina un trial.

    def info(self):
        print('rwrd_out: ' + str(self.rwrd_out))
        print('delay_out: ' + str(self.delay_out))
        print('trial_end: ' + str(self.trial_end))
        print(self.cond_0)

    def nosepoke(self, posi, posj, borra=False):
        """
        Posiciona el nosepoke en la matriz sensor. Se define con las coordenas de referencias que corresponde al centro
        de simetria.
        :param posi: coordenada de referencia (filas)
        :param posj: coordenada de referencia (columnas)
        :param borra: si es False (default) escribe (pone unos), en caso contrario, True, borra (pone ceros)
        :return: retorna en self.sensor
        """
        if borra:
            self.sensor[posi - 3:posi + 3 + 1, posj - 3:posj + 3 + 1] = 0
        else:
            self.sensor[posi - 3:posi + 3 + 1, posj - 3:posj + 3 + 1] = 1
            self.sensor[posi - 2:posi + 2 + 1, posj - 2:posj + 2 + 1] = 0

    def leverplt(self, posi, posj, borra=False):
        """
        Grafica una palanca.
        :param posi: coordenada de referencia (filas)
        :param posj: coordenada de referencia (columnas)
        :param borra: si es False (default) escribe (pone unos), en caso contrario, True, borra (pone ceros)
        :return: retorna en self.sensor
        """
        if borra:
            self.sensor[posi, posj - 2:posj + 3] = 0
            self.sensor[posi - 1:posi + 2, posj - 2] = 0
            self.sensor[posi - 1:posi + 2, posj + 2] = 0
        else:
            self.sensor[posi, posj - 2:posj + 3] = 1
            self.sensor[posi - 1:posi + 2, posj - 2] = 1
            self.sensor[posi - 1:posi + 2, posj + 2] = 1

    def bulb(self, posi, posj, off=False):
        """
        Indicador , basicamente un pixel binario
        :param posi: coordenada de referencia (filas)
        :param posj: coordenada de referencia (columnas)
        :param off: si es False (default) escribe (pone unos), en caso contrario, True, borra (pone ceros)
        :return:
        """
        if off:
            self.sensor[posi - 1:posi + 2, posj] = 0
            self.sensor[posi, posj - 1:posj + 2] = 0

        else:
            self.sensor[posi - 1:posi + 2, posj] = 1
            self.sensor[posi, posj - 1:posj + 2] = 1

    def clearframe(self):
        """
        Limpia el sensor. Pasa a cero
        :return:
        """
        self.sensor = np.zeros((self.NN, self.MM))

    def tarea_rl_init(self):
        """
        Inicializa la tarea, y su ejecución. Es necesario ejecutarlo previamente a la realización de una tarea, o puede
        emplearse como reset de la tarea.
        :return:
        """
        # Estados internos.
        self.interno_estados = [0, 1, 2, 3]
        # MAtriz de transicion
        aux = np.array([[2, 1, 3, 0, 0, 0], [0, 1, 3, 1, 1, 1], [2, 0, 3, 2, 2, 2], [3, 3, 3, 0, 3, 3]])
        self.transicion = pd.DataFrame(aux, index=self.interno_estados, columns=self.set_acciones)
        # Matriz de Reward
        aux = np.array([[0, 0, 0, -1, 0, -1], [0, -1, 0, -1, 0, 0], [-1, 0, 0, -1, 0, 0], [-1, -1, -1, 0, 0, -1]])
        self.reward = pd.DataFrame(aux, index=self.interno_estados, columns=self.set_acciones)
        self.reward.loc[0:3, 'AVA'] = 'CCHK'  # Chequea condición
        # Matriz de Delays
        self.delay = pd.DataFrame(np.ones(shape=(len(self.interno_estados), len(self.set_acciones))),
                                  index=self.interno_estados, columns=self.set_acciones)
        # Iniciamos sensor en lo que llamamos estado 0
        self.estado_caja = 0
        self.clearframe()
        self.nosepoke(int((self.NN - 1) / 2) + 1, int((self.MM - 1) / 2))
        self.leverplt(int((self.NN - 1) / 2) + 1, 17)
        self.leverplt(int((self.NN - 1) / 2) + 1, 3)
        self.tarea_rl_sched_setting()

    def tarea_rl_sched_setting(self):
        """
        Permite setear los parametros del trial (umbral y acumulador)
        Se debe ejecutar cada vez que se considera terminada la tarea.
        Esto se hace automáticamente en tarea_rl(action)
        :return:
        """
        if (self.schedule == 'CR') or (self.schedule == 'FI'):
            self.schedule_umbral = self.schedule_param
        else:
            stddev = np.floor(self.schedule_param / 3)  # desvío a partir de la media, dada por self.schedule_param
            self.schedule_umbral = stddev * np.random.randn() + self.schedule_param
        self.schedule_acumulador = 0

    def tarea_rl(self, action):
        """
        Esjecuta la tarea de acuerdo a las acciones que se toman
        :param action: acción tomada
        :return: No retorna nada particularmente, pero modifica:
        self.delay_out
        self.rwrd_out
        self.estado_caja
        """

        # Determinamos cual es el reward y la demora

        self.rwrd_out = self.reward.loc[self.estado_caja, action]
        if str(self.rwrd_out) == 'CCHK':  # Vamos a checkear si hay condicion de rwrd.
            if self.cond_0:
                self.rwrd_out = 1
            else:
                self.rwrd_out = 0
        self.delay_out = self.delay.loc[self.estado_caja, action]

        # Actualizamos el acumulador self.schedule_acumulador

        if (self.schedule == 'CR') or (self.schedule == 'RR'):  # acumula palancazos
            if action == 'PAL':  # acumula unicamente si palanquea
                if self.rwrd_izq:  # reward a Izquierda
                    if self.estado_caja == 1:  # Es un chequeo que creo no tiene sentido.
                        self.schedule_acumulador += 1
                else:
                    if self.estado_caja == 2:
                        self.schedule_acumulador += 1
        else:  # acumula tiempo
            self.schedule_acumulador = self.schedule_acumulador + self.delay_out

        # Determinamos las condiciones de rewrd de cada paradigma:
        #
        # Como cada paradigma termina siendo una comparación, o de tiempo transcurrido,
        # o de palanqueos, y la recompenza indefenctiblemente requiere previamente palanquear,
        # entonces no hace falta mirar cada paradigma. Basta con que luego de una palanca correcta
        # se llegue a la condición.

        if action == 'PAL':  # Si hay palanca, entonces hay condicionante
            if self.schedule_acumulador >= self.schedule_umbral:  # si se superó el umbral hay rwrd.
                if self.rwrd_izq:  # reward a Izquierda
                    if self.estado_caja == 1:
                        self.cond_0 = 1  # está listo el rwrd
                else:
                    if self.estado_caja == 2:
                        self.cond_0 = 1  # está listo el rwrd

        # Determinamos el nuevo estado

        self.estado_caja = self.transicion.loc[self.estado_caja][action]

        #  Graficamos el sensor
        if self.estado_caja == 0:
            self.trial_end = False  # O empieza un trial o se está en el trial.
            self.clearframe()
            self.nosepoke(int((self.NN - 1) / 2) + 1, 10)
            self.leverplt(int((self.NN - 1) / 2) + 1, 17)
            self.leverplt(int((self.NN - 1) / 2) + 1, 3)
            if self.cond_0:  # condicion de reward habilitado
                if self.rwrd_izq:  # condicion de izquierda
                    self.bulb(1, 10)
                else:
                    self.bulb(1, 10)
        elif self.estado_caja == 1:
            self.clearframe()
            self.nosepoke(int((self.NN - 1) / 2) + 1, 16)
            self.leverplt(int((self.NN - 1) / 2) + 1, 9)
            if self.cond_0:  # and self.rwrd_izq:  # condicion de reward habilitado e izquierda
                self.bulb(1, 16)
        elif self.estado_caja == 2:
            self.clearframe()
            self.nosepoke(int((self.NN - 1) / 2) + 1, 4)
            self.leverplt(int((self.NN - 1) / 2) + 1, 11)
            if self.cond_0:  # and not self.rwrd_izq:  # condicion de reward habilitado e izquierda
                self.bulb(1, 4)
        elif self.estado_caja == 3:
            self.clearframe()
            self.cond_0 = False
            self.trial_end = True  # Finaliza el trial y no empieza otro hasta que no sale del estado 3
            self.tarea_rl_sched_setting()  # resetea la tarea.

    def play(self, sched='CR', sched_par=1, rwrd_izq=False, timestep=1):
        self.schedule = sched
        self.schedule_param = sched_par
        self.rwrd_izq = rwrd_izq
        self.tarea_rl_init()
        # generar una figura
        imshow('tst', self.sensor)
        tref = time.time()

        while True:

            key = waitKey(1) & 0xFF

            if key == ord("w"):
                self.tarea_rl('AVA')
                imshow('tst', self.sensor)
                tref = time.time()
                self.info()

            if key == ord("s"):
                self.tarea_rl('RET')
                imshow('tst', self.sensor)
                tref = time.time()
                self.info()

            if key == ord("a"):
                self.tarea_rl('IZQ')
                imshow('tst', self.sensor)
                tref = time.time()
                self.info()

            if key == ord("d"):
                self.tarea_rl('DER')
                imshow('tst', self.sensor)
                tref = time.time()
                self.info()

            if key == ord("p"):
                self.tarea_rl('PAL')
                imshow('tst', self.sensor)
                tref = time.time()
                self.info()

            if key == ord("r"):
                self.tarea_rl_init()
                imshow('tst', self.sensor)
                tref = time.time()
                self.info()

            if time.time() - tref > timestep:
                self.tarea_rl('NAM')
                tref = time.time()

            if key == ord("c"):
                break

        print('FIN')


class Cortex:
    """
    Esta clase se comporta de cierta forma como la corteza. Conforma la etapa que adquiere datos sensoriales, emite la
    acción que realizará, y registra la recompenza.
    Entre las cosas que hace es organizar
    """
