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

def partir(t, p, u, pivotF):
    pivote = pivotF(t, p, u)
    for i in range(p+1, u+1):
        if t[i] <= t[p]:
            pivote += 1
            t[i], t[pivote] = t[pivote], t[i]
    t[pivote], t[p] = t[p], t[pivote]
    return pivote

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

#Ejecucion
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