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

##################################################
# PARTE 3 - Grafos, Dijkstra, Floyd-Warshall y TGF
##################################################

def randMatrPosWGraph(nNodes, sparseFactor, maxWeight=50):
  matriz = np.zeros(shape=(nNodes,nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      if i!=j and sparseFactor>=np.random.random_sample():
        matriz[i,j]=random.randrange(maxWeight)+1
      elif i!=j:
        matriz[i,j]=np.inf
  return matriz

def cuentaRamas(mG):
  res=0
  for i in range(len(mG)):
    for j in range(len(mG)):
      if mG[i,j]!=np.inf and i!=j:
        res=res+1
  return res

def fromAdjM2Dict(mG):
  res = {}
  for i in range(len(mG)):
    alist = []
    for j in range(len(mG)):
      if i!=j and mG[i,j]!=np.inf:
       alist.append((j, mG[i,j]))
    res.update({ i : alist})
  return res

def fromDict2AdjM(dG):
  matriz = np.zeros(shape=(len(dG), len(dG)))
  for i in range(len(dG)):
    for j in range(len(dG)):
      if i!=j:
        matriz[i,j]=np.inf
  for i in range(len(dG)):
    alist = dG.get(i)
    for j in range(len(alist)):
      tupla = alist[j]
      matriz[i,tupla[0]]=tupla[1]
  return matriz

def dijkstraM(mG, u=0):
  previos = np.zeros(len(mG))
  dist = np.zeros(len(mG))
  vistos = [False for x in range(len(mG))]
  pq = Queue.PriorityQueue()

  for i in range(len(mG)):
      dist[i]=np.inf
      previos[i]=-1
  dist[u]=0
  pq.put((0,u))

  while pq.empty()==False:
    tupla = pq.get()
    if vistos[int(tupla[1])]==False:
      vistos[int(tupla[1])] = True
      for i in range(len(mG)):
        if dist[i]>dist[tupla[1]]+mG[tupla[1],i]:
          previos[i]=tupla[1]
          dist[i]=mG[tupla[1],i]+dist[tupla[1]]
          pq.put((dist[i], i))

  return previos,dist
    
def dijkstraD(dG, u=0):
  previos = np.zeros(len(dG))
  dist = np.zeros(len(dG))
  vistos = [False for x in range(len(dG))]
  pq = Queue.PriorityQueue()

  for i in range(len(dG)):
      dist[i]=np.inf
      previos[i]=-1
  dist[u]=0
  pq.put((0,u))

  while pq.empty()==False:
    tupla = pq.get()
    if vistos[int(tupla[1])]==False:
      vistos[int(tupla[1])] = True
      alist=dG.get(tupla[1])
      for i in range(len(alist)):
        nodo=alist[i][0]
        if dist[nodo]>dist[tupla[1]]+alist[i][1]:
          previos[nodo]=tupla[1]
          dist[nodo]=alist[i][1]+dist[tupla[1]]
          pq.put((dist[nodo], nodo))

  return previos, dist
  
def timeDijkstraM(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25):
  res, tmp = [], []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrPosWGraph(i, sparseFactor)
      for k in range(len(matriz)):
        ti = time.time()
        dijkstraM(matriz,k)
        tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def timeDijkstraD(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25):
  res, tmp = [], []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrPosWGraph(i, sparseFactor)
      dic=fromAdjM2Dict(matriz)
      for k in range(len(matriz)):
        ti = time.time()
        dijkstraD(dic,k)
        tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def n2lognvalues(sizeIni, sizeFin, step):
    l=[]
    for i in range(sizeIni, sizeFin, step):
        l.append(i*i*log(i,10))
    return l
  
def fitPlotDijkstra(lM, lD, func2fit, nNodesIni, nNodesFin, step):
    regr = linear_model.LinearRegression()
    x=np.array(func2fit(nNodesIni, nNodesFin, step))
    yM=np.array(lM)
    yD=np.array(lD)

    fit=regr.fit(yM[:,np.newaxis], x)
    valM=regr.predict(yM[:,np.newaxis])

    fit=regr.fit(yD[:,np.newaxis], x)
    valD=regr.predict(yD[:,np.newaxis])

    blue_patch = mpatches.Patch(color='red', label='Dijkstra Matriz => *')
    green_patch = mpatches.Patch(color='blue', label='Dijkstra Diccionario => .')
    red_patch = mpatches.Patch(color='green', label='N2logN')
    plt.legend(handles=[blue_patch, green_patch, red_patch],loc=4)

    plt.plot(valM,'*',color='red')
    plt.plot(valD, '.',color='blue')
    plt.plot(x,color='green')
    plt.ylabel("Tiempo")
    plt.title("Dijkstra Matriz VS Dijkstra Matriz with Fit N2logN")
    plt.show()
    return

def dijkstraMAllPairs(mG):
  matriz = np.zeros(shape=(len(mG), len(mG)))
  for i in range (len(mG)):
    aux = dijkstraM(mG, i)
    matriz[i] = aux[1]
  return matriz

def floydWarshall0(mG):
  tam=len(mG)
  d = np.zeros(shape=(len(mG), len(mG), len(mG)+1))
  d[:, :, 0] = copy.copy(mG)

  for k in range(1, tam):
    for i in range(tam):
      for j in range(tam):
        t = d[i, k, k-1] + d[k, j, k-1]
      	d[i, j, k] = min (t, d[i, j, k-1])  	
  return d[:, :, tam-1]

def floydWarshall(mG): #Pisando la matriz, mucho mas rapido
  tam=len(mG)
  m = copy.copy(mG)

  for k in range(tam):
    for i in range(tam):
      for j in range(tam):
        if m[i, j] > m[i, k] + m[k, j]:
          m[i, j] = m[i, k] + m[k, j]
  return m

def timeDijkstraMAllPairs(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25):
  res, tmp = [], []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrPosWGraph(i, sparseFactor)
      ti = time.time()
      dijkstraMAllPairs(matriz)
      tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def timeFloydWarshall(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25):
  res, tmp = [], []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrPosWGraph(i, sparseFactor)
      ti = time.time()
      floydWarshall(matriz)
      tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def timeFloydWarshall2(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25):
  res, tmp = [], []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrPosWGraph(i, sparseFactor)
      ti = time.time()
      floydWarshall2(matriz)
      tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

####################################
# PARTE 4 - The Trivial Graph Format
####################################
def mG2TGF(mG, fName):
  with open(fName, "w") as f:
    for i in range(len(mG)):
      f.write(str(i)+"\n")
    f.write("#\n")
    for i in range(len(mG)):
      for j in range(len(mG)):
        if mG[i,j]!=np.inf and i!=j:
          f.write(str(i)+" "+str(j)+" "+str(mG[i,j])+"\n")
  return

def TGF2mG(fName):
  with open(fName, "r") as f:
    size=0
    while (f.readline()[0] != "#"):
      size=size+1
    matriz = np.zeros(shape=(size, size))
    for i in range (size):
      for j in range (size):
        if i!=j:
          matriz[i, j] = np.inf
    for line in f:
      val = line.split( )
      matriz[val[0], val[1]] = val[2]
  return matriz

def dG2TGF(dG, fName):
  with open(fName, "w") as f:
    for i in range(len(dG)):
      f.write(str(i)+"\n")
    f.write("#\n")
    for i in range(len(dG)):
      alist = dG.get(i)
      for j in range(len(alist)):
        f.write(str(i)+" "+str(alist[j][0])+" "+str(alist[j][1])+"\n")
  return

def TGF2dG(fName):
  with open(fName, "r") as f:
    size=0
    res = {}
    while (f.readline()[0] != "#"):
      alist = []
      res.update({ size : alist})
      size=size+1
    for line in f:
      val = line.split( )
      alist = res.get(int(val[0]))
      if alist==None:
        alist = [(int(val[1]), float(val[2]))]
      else:
        alist.append((int(val[1]), float(val[2])))
      res.update({ int(val[0]) : alist})     
  return res
'''
  #Ejecucion
def main(argv):
  fName = argv[1]
  dG = TGF2dG(fName)
  matriz = fromDict2AdjM(dG)
  res = floydWarshall(matriz)
  if len(res)<=8:
  	print (res)
  	return
  res2 = np.zeros(shape=(8, 8))
  for i in range(8):
  	for j in range(8):
  		res2[i, j] = res[i, j]
  print (res2)
  return

if __name__ == "__main__":
  main(sys.argv)
'''
dG=TGF2dG("dg.tgf")
print dG
mG=TGF2mG("dg.tgf")
print mG

print dijkstraM(mG)
print dijkstraD(dG)


lm=timeDijkstraM(10, 10, 100, 10)
ld=timeDijkstraD(10, 10, 100, 10)
print lm
print ld

fitPlotDijkstra(lm, ld,n2lognvalues, 10, 100, 10)

