# -*- encoding: utf-8 -*-

#Imports
import sys
import random
import string
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import linear_model
from math import log
import Queue
import copy

##############
##############
# PRACTICA 1 #
##############
##############

##################
# PARTE 2 - QS
##################
def permutacion(sizeP):
  return np.random.permutation(sizeP)

def checkPerms(numPerms, sizeP):
  l=[]
  for i in range (numPerms):
      l.append(permutacion(sizeP))
  plt.show(plt.hist(l.pop()))
  return

def firstP (t, p, u):
  return p

def qs(t, p=0, u=None, pivotF=firstP):
    if u is None:
        u = len(t) - 1
    if p >= u:
        return t
    pivote = partir(t, p, u, pivotF)
    qs(t, p, pivote-1, pivotF)
    qs(t, pivote+1, u, pivotF)
    return t

def qs2(t, p=0, u=None):
  if u is None:
    u = len(t) - 1
  return np.concatenate((t[0:p], qs_real2(t[p:u+1]), t[u+1:t.size]))

def qs_real2(t):
  if len(t)<2: return t #Nada que ordenar
  menor, igual, mayor = [], [], []
  pivote = t[0]
  for i in t:
    if i < pivote:
      menor.append(i)
    elif i == pivote:
      igual.append(i)
    else:
      mayor.append(i)
  return qs_real2(menor)+igual+qs_real2(mayor)

def qs3(t, p=0, u=None, pivotF=firstP):
    if u is None:
        u = len(t) - 1
    while p < u:
        pivote = partir(t, p, u, pivotF)
        qs3(t, p, pivote, pivotF)
        p = pivote + 1
    return t

def qs4(t, p=0, u=None):
  if u is None:
    u = len(t) - 1
  return np.concatenate((t[0:p], qs_real4(t[p:u+1]), t[u+1:t.size]))

def qs_real4(t):
    if len(t)<2: return t #Nada que ordenar
    pivote = t[0]
    menor = qs_real4([x for x in t[1:] if x < pivote])
    mayor = qs_real4([x for x in t[1:] if x >= pivote])
    return menor + [pivote] + mayor

def timeSort(sortM, nPerms, sizeIni, sizeFin, step):
  res, tmp = [], []
  for i in range(sizeIni, sizeFin, step):
    for j in range(nPerms):
      alist=permutacion(i)
      ti = time.time()
      sortM(alist)
      tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def nlognvalues(sizeIni, sizeFin, step):
    l=[]
    for i in range(sizeIni, sizeFin, step):
      l.append(i*log(i,10))
    return l

def fitPlot(l, func2fit, nNodesIni, nNodesFin, step):
    regr = linear_model.LinearRegression()
    x=np.array(func2fit(nNodesIni, nNodesFin, step))
    y=np.array(l)

    fit=regr.fit(y[:,np.newaxis], x)
    val=regr.predict(y[:,np.newaxis])
    blue_patch = mpatches.Patch(color='blue', label='Quicksort')
    red_patch = mpatches.Patch(color='red', label='NlogN')
    plt.legend(handles=[blue_patch, red_patch])

    plt.plot(val,'*',color='blue')
    plt.plot(x,color='red')
    plt.ylabel("Tiempo")
    plt.title("Quicksort Fit NlogN")
    plt.show()
    return

##############
##############
# PRACTICA 3 #
##############
##############

####################################
# PARTE 2 - El problema de selección
####################################

# Funciones basicas

def partir(t, p, u, pivotF):
  """
  Realiza la reordenación de la tabla en función de un pivote
  :param t: tabla a reorganizar
  :param p: primera posición de la tabla
  :param u: última posición de la tabla
  :return: pivote obtenido
  """
  pivote = pivotF(t, p, u)
  k=t[pivote]
  t[p], t[pivote] = t[pivote], t[p]
  pivote=p
  for i in range(p+1, u+1):
    if t[i] < k:
      pivote += 1
      t[i], t[pivote] = t[pivote], t[i]
  t[pivote], t[p] = t[p], t[pivote]
  return pivote

def pivotP(t, p, u):
  """
  Devuelve la primera posición de la tabla como pivote
  :param t: tabla
  :param p: primera posición de la tabla
  :param u: última posición de la tabla
  :return: pivote
  """
  return p

def pivot5_value(t, p, u):
  """
  Devuelve el valor de la mediana de la tabla como pivote
  :param t: tabla
  :param p: primera posición de la tabla
  :param u: última posición de la tabla
  :return: pivote valor
  """
  ultimo = -1
  medianas = []
  if len(t[p:u+1]) <= 5:
    copia = [a for a in t[p:u+1]]
    copia.sort()
    return copia[len(copia)//2]
  for i in range(p, u+1, 5):
    ultimo = i+5
    if i+4 > u:
      ultimo=u+1
    t[i:ultimo].sort()
    medianas.append ( t[i:ultimo] [len(t[i:ultimo])//2] )
  return qselect(medianas, 0, len(medianas)-1, len(medianas)//2, pivot5)

def pivot5(t, p, u):
  """
  Devuelve la mediana de la tabla como pivote
  :param t: tabla
  :param p: primera posición de la tabla
  :param u: última posición de la tabla
  :return: pivote
  """
  x=np.array(t[0:len(t)])
  result = np.where(x == pivot5_value(t, p, u))[0][0]
  return result

def qselect(t, p, u, k, pivotF = pivotP):
  """
  Encuentra el k esimo número de una tabla independientemente de su ordenación
  :param t: tabla a buscar
  :param p: primera posición de la tabla
  :param u: última posición de la tabla
  :param k: posición de la tabla cuyo valor quiero saber si estubiese ordenada
  :param pivotF: función para buscar pivote, por defecto pivotP
  :return: valor k esimo
  """
  if k > u-p or k < 0:
    return None
  m = partir(t, p, u, pivotF)
  if k == m-p:
    return t[m]
  elif k < m-p:
    return qselect(t, p, m-1, k, pivotF)
  else: #k > m-p+1:
    return qselect(t, m+1, u, k-(m-p+1), pivotF)

# Timing QuickSelect and QuickSort

def time_qselect_ave(nPerms, sizeIni, sizeFin, step, pivotF=pivotP):
  """
  Mide los tiempos de ejecucion medios de quickselect
  :param nPerms: numero de repeticiones de cada prueba
  :param sizeIni: tamaño inicial de las pruebas
  :param sizeFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :param pivotF: función que devuelve pivote
  :return: lista del tiempo medio de las repeticiones de cada prueba
  """
  res, tmp = [], []
  for i in range(sizeIni, sizeFin+1, step):
    for j in range(nPerms):
      alist=permutacion(i)
      ti = time.clock()#####################################################################
      qselect(alist, 0, len(alist)-1, len(alist)/2, pivotF)
      tmp.append(time.clock()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def time_qselect_worst(nPerms, sizeIni, sizeFin, step, pivotF=pivotP):
  """
  Mide los tiempos de ejecucion peores de quickselect
  :param nPerms: numero de repeticiones de cada prueba
  :param sizeIni: tamaño inicial de las pruebas
  :param sizeFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :param pivotF: función que devuelve pivote
  :return: lista del tiempo peor de las repeticiones de cada prueba
  """
  res, tmp = [], []
  for i in range(sizeIni, sizeFin+1, step):
    for j in range(nPerms):
      alist=permutacion(i)
      ti = time.clock()#####################################################################
      qselect(alist, 0, len(alist)-1, len(alist)/2, pivotF)
      tmp.append(time.clock()-ti)
    res.append(max(tmp))
    tmp = []
  return res

def timeSort_worst(sortM, nPerms, sizeIni, sizeFin, step, pivot_f=None):
  """
  Mide los tiempos de ejecucion peor de un algoritmo de ordenación
  :param sortM: algoritmo de ordenacion a utilizar
  :param nPerms: numero de repeticiones de cada prueba
  :param sizeIni: tamaño inicial de las pruebas
  :param sizeFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :param pivotF: función que devuelve pivote
  :return: lista del tiempo peor de las repeticiones de cada prueba
  """
  res, tmp = [], []
  for i in range(sizeIni, sizeFin, step):
    for j in range(nPerms):
      alist=permutacion(i)
      alist=sorted(alist)
      ti = time.time()
      sortM(alist, 0, len(alist)-1, pivot_f)
      tmp.append(time.time()-ti)
    res.append(max(tmp))
    tmp = []
  return res

def nvalues(sizeIni, sizeFin, step):
  """
  Genera valores N
  :param sizeIni: inicio eje x
  :param sizeFin: fin eje x
  :param step: salto entre valores
  :return: lista con los valores para y
  """
  l=[]
  for i in range(sizeIni, sizeFin+1, step):
    l.append(i)
  return l

def n2values(sizeIni, sizeFin, step):
  """
  Genera valores N^2
  :param sizeIni: inicio eje x
  :param sizeFin: fin eje x
  :param step: salto entre valores
  :return: lista con los valores para y
  """
  l=[]
  for i in range(sizeIni, sizeFin+1, step):
    l.append(i*i)
  return l

def fitPlotQS(p, func2fitP, nNodesIni, nNodesFin, step):
  """
  Fit a los tiempos teóricos de los tiempos de ejecucion de quickselect
  :param p: array de tiempos para quickselect
  :param func2fitP: funcion para hacer el fit
  :param nNodesIni: tamaño inicial de las pruebas
  :param nNodesFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :return: None
  """
  regr = linear_model.LinearRegression()
  xP=np.array(func2fitP(nNodesIni, nNodesFin, step))
  yP=np.array(p)

  fit=regr.fit(yP[:,np.newaxis], xP)
  valM=regr.predict(yP[:,np.newaxis])

  plt.plot(valM, '*',label='Pivot 5')
  plt.plot(xP,label='Pivot 5 teorica')
  plt.legend(loc=4)
  plt.ylabel("Tiempo")
  plt.xlabel("Vertices")
  plt.title("QS Pivot5 with Fit teoricas")
  plt.show()
  return

#Ejecucion
'''
def main(argv):
  nPerms = int(argv[2])
  sizeIni = int(argv[4])
  sizeFin = int(argv[6])
  step = int(argv[8]) 
  res = timeSort(qs2, nPerms, sizeIni, sizeFin, step)
  for i in range(len(res)):
    print ("tamaño:\t%i t medio:\t%s" % (sizeIni, res[i]))
    sizeIni = sizeIni + step
  return

if __name__ == "__main__":
  main(sys.argv)
'''


