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
import numpy.ma as ma

##############
##############
# PRACTICA 1 #
##############
##############

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
      floydWarshall0(matriz)
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



##############
##############
# PRACTICA 2 #
##############
##############


#######################################################
# PARTE 1 - Grafos no dirigidos y TAD Conjunto Disjunto
#######################################################

### Gragos no dirigidos

def randMatrUndPosWGraph(nNodes, sparseFactor=0.5, maxWeight=50.):
  """
  Genera una matriz que representa un grafo de nNodes con posibilidad de rama sparseFactor y un peso maximo indicado
  :param nNodes: numero de nodos
  :param sparseFactor: probabilidad de rama presente
  :param maxWeight: peso máximo de las ramas
  :return: matriz creada
  """
  matriz = np.zeros(shape=(nNodes,nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      if j>i:
        if i!=j and sparseFactor>=np.random.random_sample():
          matriz[i,j]=random.randrange(maxWeight)+1
        elif i!=j:
          matriz[i,j]=np.inf
      elif j<i:
      	matriz[i, j] = matriz [j, i]
  return matriz

def checkUndirectedM(mG):
  """
  Comprueba si un grafo representado en una matriz es dirigido o no
  :param mG: matriz que representa el grafo
  :return: True si es no dirigido, False si no
  """
  #Creamos una mascara y la aplicamos a la 1 matriz
  mascara = np.zeros(shape=(len(mG),len(mG)))
  for i in range(len(mG)):
    mascara[i:len(mG):1, i]=1
  m1 = copy.copy(ma.array(mG, mask = mascara))

  #Hallamos la t. de la 2 y aplicamos la mascara
  m2 = np.zeros(shape=(len(mG),len(mG)))
  for i in range(len(mG)):
    m2[i, i:len(mG):1] = mG[i:len(mG):1, i]

  m2 = copy.copy(ma.array(m2, mask = mascara))

  return np.all(np.equal(m1, m2))

def checkUndirectedM2(mG):
  """
  Comprueba si un grafo representado en una matriz es dirigido o no
  :param mG: matriz que representa el grafo
  :return: True si es no dirigido, False si no
  """
  #Para cada fila comprobamos si es igual a una columna
  for i in range(len(mG)):
    if np.array_equal( mG[i:len(mG):1, i], mG[i, i:len(mG):1] ) == False: #Indice i nos ahorra comprobaciones repetidas
  	  return False
  return True

def checkUndirectedM3(mG):
  """
  Comprueba si un grafo representado en una matriz es dirigido o no
  :param mG: matriz que representa el grafo
  :return: True si es no dirigido, False si no
  """
  a = np.tril(mG, 0) #Superior
  b = np.triu(mG, 0) #Inferior
  return np.all(np.equal(a, b.transpose() )) #Comprobamos igualdad de sup y t. inf

def checkUndirectedD(dG):
  """
  Comprueba si un grafo representado en un diccionario es dirigido o no
  :param dG: diccionario que reprenseta el grafo
  :return: True si es no dirigido, False si no
  """
  flag=True
  for i in range(len(dG)):
    alist = dG.get(i)
    #cogemos todos los edges de i
    for j in range(len(alist)):
      if flag==False:
        return False
      flag=False
      valor = alist[j][1]
      l = dG.get(alist[j][0])
      #comprobamos si existe conexion y peso correspondiente con i
      for k in range(len(l)):
        nodo = l[k]
        if nodo[0]==i and nodo[1]==valor:
          flag=True
          break
  return True

### Conjunto disjunto

def initCD(N):
  """
  Inicia un conjunto disjunto
  :param N: tamaño
  :return: conjunto disjunto creado
  """
  cd = np.arange(N)
  cd.fill(-1)
  return cd

def union(rep1, rep2, pS):
  """
  Reliza la union de dos arboles de un cd
  :param rep1: representante del primer arbol
  :param rep2: representante del segundo arbol
  :param pS: conjunto disjunto
  :return: representante de la union
  """
  #Asumimos que aqui llegan los nodos raiz
  if pS[rep2] < pS[rep1]: #Arbol de rep2 mas profundo
    pS[rep1] = rep2
    return rep2
  elif pS[rep2] > pS[rep1]: #Arbol de rep1 mas profundo
    pS[rep2] = rep1
    return rep1
  else: #Igual profundidad
    pS[rep2] = rep1
    pS[rep1] = pS[rep1] - 1
    return rep1

def find(ind, pS, flagCC=True):
  """
  Encuentra el representante de un arbol en un cd
  :param ind: elemento del que se desea el representante
  :param pS: conjunto disjunto
  :param flagCC: indica si se realiza compresion de caminos
  :return: representante del elemento deseado
  """
  z=ind
  while pS[z] >= 0:
    z = pS[z]
  #Compresion de caminos
  if flagCC==True:
    while pS[ind] >= 0:
      y = pS[ind]
      pS[ind] = z
      ind = y
  #Devolvemos el nodo raiz
  return z

#######################################
# PARTE 2 - Arboles abarcadores mınimos
#######################################

### Algoritmo de kruscal

def insertPQ(dG, Q):
  """
  Inserta las ramas origen<destino de un grafo en una PQ
  :param dG: diccionario que representa un grafo
  :param Q: cola de prioridad iniciada
  :return: cola de prioridad con los elementos
  """
  for i in range(len(dG)):
    alist = dG.get(i)
    for j in range(len(alist)):
      if i<alist[j][0]:
        Q.put((alist[j][1], i, alist[j][0]))
  return Q

def insertPQmatrix(mG, Q):
  """
  Inserta las ramas origen<destino de un grafo en una PQ
  :param mG: matriz que representa un grafo
  :param Q: cola de prioridad iniciada
  :return: cola de prioridad con los elementos
  """
  for i in range(len(mG)):
    for j in range(len(mG)):
      if i<j:
        Q.put((mG[i, j], i, j))
  return Q

def kruskallist(mG, flagCC=True):
  """
  Realiza el algoritmo de kruscal
  :param mG: matriz que representa el grafo
  :param flagCC: indica si se realiza compresion de caminos
  :return: lista con las ramas seleccionadas
  """
  res = []
  pq = Queue.PriorityQueue()
  pq = insertPQmatrix(mG, pq)
  cd=initCD(len(mG))
  while pq.empty() == False:
    camino = pq.get()
    x = find(camino[1], cd, flagCC)
    y = find(camino[2], cd, flagCC)
    if x != y:
      res.append((camino[1], camino[2]))
      union(x, y, cd)
  return res;

def kruskal(dG, flagCC=True):
  """
  Realiza el algoritmo de kruscal
  :param dG: diccionario que representa el grafo
  :param flagCC: indica si se realiza compresion de caminos
  :return: diccionario con las ramas seleccionadas
  """
  res = {}
  pq = Queue.PriorityQueue()
  pq = insertPQ(dG, pq)
  cd=initCD(len(dG))
  while pq.empty() == False:
    camino = pq.get()
    x = find(camino[1], cd, flagCC)
    y = find(camino[2], cd, flagCC)
    if x != y:
      if camino[1] in res:
        aux=res.get(camino[1])
        aux.append((camino[2],camino[0]))
        res.update({camino[1]: aux})
      else:
        res.update({camino[1]: [(camino[2],camino[0])]})
      if camino[2] in res:
        aux=res.get(camino[2])
        aux.append((camino[1],camino[0]))
        res.update({camino[2]: aux})
      else:
         res.update({camino[2]: [(camino[1],camino[0])]})
      union(x, y, cd)
  return res;

### Coste de Kruscal

def timeKruskal(nGraphs, nNodesIni, nNodesFin, step, sparseFactor, flagCC):
  """
  Mide los tiempos de ejecucion de Kruscal
  :param nGraphs: numero de repeticiones de cada prueba
  :param nNodesIni: tamaño inicial de las pruebas
  :param nNodesFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :param sparseFactor: probabilidad de rama presente
  :param flagCC: indica si se realiza compresion de caminos
  :return: lista del tiempo medio de las repeticiones de cada prueba
  """
  res = []
  tmp = []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrUndPosWGraph(i, sparseFactor)
      dic=fromAdjM2Dict(matriz)
      for k in range(len(matriz)):
        ti = time.time()
        kruskal(dic,flagCC)
        tmp.append(time.time()-ti)
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res

def elognvalues(sizeIni, sizeFin, step, sparseFactor):
  """
  Genera valores ElogN
  :param sizeIni: inicio eje x
  :param sizeFin: fin eje x
  :param step: salto entre valores
  :param sparseFactor: probabilidad de rama presente
  :return: lista con los valores para y
  """
  l=[]
  for i in range(sizeIni, sizeFin, step):
    l.append(sparseFactor*i*i*log(i,10))
  return l

def kruskal02(dG, flagCC=True):
  """
  Realiza el algoritmo de kruscal midiendo tiempos del while
  :param dG: diccionario que representa el grafo
  :param flagCC: indica si se realiza compresion de caminos
  :return: tiempo de ejecucion del while
  """
  res = {}
  pq = Queue.PriorityQueue()
  pq = insertPQ(dG, pq)
  cd=initCD(len(dG))
  ti = time.time()
  while pq.empty() == False:
    camino = pq.get()
    x = find(camino[1], cd, flagCC)
    y = find(camino[2], cd, flagCC)
    if x != y:
      if camino[1] in res:
        aux=res.get(camino[1])
        aux.append((camino[2],camino[0]))
        res.update({camino[1]: aux})
      else:
        res.update({camino[1]: [(camino[2],camino[0])]})
      if camino[2] in res:
        aux=res.get(camino[2])
        aux.append((camino[1],camino[0]))
        res.update({camino[2]: aux})
      else:
         res.update({camino[2]: [(camino[1],camino[0])]})
      union(x, y, cd)
  tmp = (time.time()-ti)
  return tmp;

def timeKruskal02(nGraphs, nNodesIni, nNodesFin, step, sparseFactor, flagCC):
  """
  Mide los tiempos de ejecucion del while de Kruscal
  :param nGraphs: numero de repeticiones de cada prueba
  :param nNodesIni: tamaño inicial de las pruebas
  :param nNodesFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :param sparseFactor: probabilidad de rama presente
  :param flagCC: indica si se realiza compresion de caminos
  :return: lista del tiempo medio de las repeticiones de cada prueba
  """
  res = []
  tmp = []
  for i in range(nNodesIni, nNodesFin, step):
    for j in range(nGraphs):
      matriz=randMatrUndPosWGraph(i, sparseFactor)
      dic=fromAdjM2Dict(matriz)
      for k in range(len(matriz)):
        tmp.append(kruskal02(dic,flagCC))
    res.append(reduce(lambda x, y: x + y, tmp) / len(tmp))
    tmp = []
  return res


def fitPlotKruskal(lM, lD, func2fit, nNodesIni, nNodesFin, step, sparseFactor):
  """
  Fit a los tiempos teóricos de los tiempos de ejecucion de Kruscal normal y kruskal para el bucle while  
  :param lM: array de tiempos para kruskal normal
  :param lD: array de tiempos para kruskal para el bucle while
  :param func2fit: funcion para hacer el fit
  :param nNodesIni: tamaño inicial de las pruebas
  :param nNodesFin: tamaño final de las pruebas
  :param step: aumento de tamaño entre pruebas
  :param sparseFactor: probabilidad de rama presente
  :return: None
  """
  regr = linear_model.LinearRegression()
  x=np.array(func2fit(nNodesIni, nNodesFin, step, sparseFactor))
  yM=np.array(lM)
  yD=np.array(lD)

  fit=regr.fit(yM[:,np.newaxis], x)
  valM=regr.predict(yM[:,np.newaxis])

  fit=regr.fit(yD[:,np.newaxis], x)
  valD=regr.predict(yD[:,np.newaxis])

  plt.plot(valM, '.',label='Kruskal')
  plt.plot(valD, '*',label='Kruskal PQ')
  plt.plot(x,label='N2logN')
  plt.legend(loc=4)
  plt.ylabel("Tiempo")
  plt.xlabel("Vertices")
  plt.title("Kruskal VS kruskal PQ with Fit |E|log|V|")
  plt.show()
  return

def F2elognMas2nMasevalues(sizeIni, sizeFin, step, sparseFactor):
  """
  Genera valores 2ElogN+2N+E
  :param sizeIni: inicio eje x
  :param sizeFin: fin eje x
  :param step: salto entre valores
  :param sparseFactor: probabilidad de rama presente
  :return: lista con los valores para y
  """
  l=[]
  for i in range(sizeIni, sizeFin, step):
    l.append((2*sparseFactor*i*i*log(i,10))+(2*i)+(sparseFactor*i*i))
  return l

def F2eMas1lognMas2nvalues(sizeIni, sizeFin, step, sparseFactor):
  """
  Genera valores (2E+1)logN+2N
  :param sizeIni: inicio eje x
  :param sizeFin: fin eje x
  :param step: salto entre valores
  :param sparseFactor: probabilidad de rama presente
  :return: lista con los valores para y
  """
  l=[]
  for i in range(sizeIni, sizeFin, step):
    l.append((((2*sparseFactor*i*i)+1)*log(i,10))+(2*i))
  return l

########################
# PARTE 2 - BP, OT y DMs
########################

### Busqueda en profundidad
def incAdy(dG):
  """
  Devuelve listas de incidencia, adyacencia e inicia la lista de previos de un grafo
  :param dG: diccionario que representa el grafo
  :return: lista incidencia, lista adyacencia, lista previos
  """
  inc = []
  ady = []
  previos = []
  for i in range(len(dG)):
    inc.append(0)
    previos.append(-1)
  for i in range(len(dG)):
    alist = dG.get(i)
    ady.append(len(alist))
    for j in range(len(alist)):
      inc[alist[j][0]]=inc[alist[j][0]]+1
  return inc, ady, previos

def drBP(dG, u=0):
  """
  Realiza BP completo para un grafo
  :param dG: diccionario que representa el grafo
  :param u: nodo de inicio
  :return: lista inicios exploracion, lista fines exploracion, lista padres
  """
  vistos = [False for x in range(len(dG))]
  padres = [-1 for x in range(len(dG))]
  ini = [-1 for x in range(len(dG))]
  fin = [-1 for x in range(len(dG))]
  t = 1
  t = BP(u, dG, vistos, padres, ini, fin, t)
  for i in range(len(dG)):
    if vistos[i]==False:
      t = BP(i, dG, vistos, padres, ini, fin, t)
  return ini, fin, padres

def BP(u, dG, vistos, padres, ini, fin, t):
  """
  Realiza BP desde un nodo
  :param u: nodo de inicio
  :param dG: diccionario que representa el grafo
  :param vistos: lista con los nodos ya visitados
  :param padres: lista con los padres de los nodos
  :param ini: lista inicios exploracion
  :param fin: lista fines exploracion
  :param t: contador de tiempos para ini y fin
  :return: contador de tiempos para ini y fin
  """
  ini[u]=t
  t=t+1
  vistos[u] = True
  alist = dG.get(u)
  for i in range(len(alist)):
    nodo = alist[i][0]
    if vistos[nodo] == False:
      padres[nodo] = u
      t = BP(nodo, dG, vistos, padres, ini, fin, t)
  fin[u]=t
  t=t+1
  return t

### DAGs y OT

def clasificaRamas(u, v, padres, ini, fin):
  """
  Clasifica una rama
  :param u: vertice inicio
  :param v: vertice fin
  :param padres: lista de padres de los nodos
  :param ini: lista inicios exploracion
  :param fin: lista fines exploracion
  :return: tipo de rama: T (arbol), D (descendente), A (ascendente) o C (cruce)
  """
  if padres[v] == u:
    return "T"
  elif padres[v]!=u and ini[u]<ini[v] and fin[u]==np.inf:
    return "D"
  elif padres[v]!=u and ini[v]<ini[u] and fin[v]==np.inf:
    return "A"
  return "C"


def drBPasc(dG, u=0):
  """
  Detector de ramas ascendentes para todo el grafo
  :param dG: diccionario que representa el grafo
  :param u: nodo inicial
  :return: lista de ramas ascendentes
  """
  vistos = [False for x in range(len(dG))]
  padres = [-1 for x in range(len(dG))]
  ini = [np.inf for x in range(len(dG))]
  fin = [np.inf for x in range(len(dG))]
  asc = []
  t = 1
  t = BPasc(u, dG, vistos, padres, ini, fin, t, asc)
  for i in range(len(dG)):
    if vistos[i]==False:
      t = BPasc(i, dG, vistos, padres, ini, fin, t, asc)
  return asc

def BPasc(u, dG, vistos, padres, ini, fin, t, asc):
  """
  Detector de ramas ascendente desde un nodo
  :param u: nodo inicial
  :param dG: diccionario que representa el grafo
  :param vistos: lista con los nodos ya visitados
  :param padres: lista con los padres de los nodos
  :param ini: lista inicios exploracion
  :param fin: lista fines exploracion
  :param t: contador de tiempos para ini y fin
  :param asc: lista de ramas acendentes
  :return: contador de tiempos para ini y fin
  """
  ini[u]=t
  t=t+1
  vistos[u] = True
  alist = dG.get(u)
  for i in range(len(alist)):
    nodo = alist[i][0]
    if(clasificaRamas(u, nodo, padres, ini, fin) == "A"):
      asc.append((u, nodo))
    if vistos[nodo] == False:
      padres[nodo] = u
      t = BPasc(nodo, dG, vistos, padres, ini, fin, t, asc)
  fin[u]=t
  t=t+1
  return t

def DAG(dG):
  """
  Indica si un grafo es un DAG
  :param dG: diccionario que representa el grafo
  :return: True si es un DAG, False si no
  """
  return drBPasc(dG)==[]

def OT(dG):
  """
  Realiza OT para un grafo
  :param dG: diccionario que representa el grafo
  :return: lista de nodos ordenados topológicamente
  """
  res=[]
  if DAG(dG):
    inc = incAdy(dG)[0]
    for i in range(len(inc)):
      if inc[i]==0:
        #Empezamos desde el primer nodo inc=0 encontrado
        fin = drBP(dG, i)[1]
        nodoMax, finMax = -1, -1
        for j in range(len(dG)):
          for k in range(len(fin)):
            if finMax<fin[k]:
              finMax=fin[k]
              nodoMax=k
          res.append(nodoMax)
          fin[nodoMax] = -1
          finMax, nodoMax = -1, -1
        return res
  return res

### Dist. Min. en DAG con 1 solo nodo fuente

def distMinSingleSourceDAG(dG):
  """
  Calcula la distancia minima de un nodo al resto de un grafo si es un DAG y ese nodo es el unico con incidencia 0
  :param dG: diccionario que representa el grafo
  :return: lista de distancias, lista de previos
  """
  padres = [-1 for x in range(len(dG))]
  distancias = [np.inf for x in range(len(dG))]
  if DAG(dG)==False:
    return distancias, padres
  inc = incAdy(dG)[0]
  count = 0
  for i in inc:
    if i==0:
      count = count+1
  if count==0:
    print "Aviso: No hay nodo fuente"
    return distancias, padres
  if count>1:
    print "Aviso: Hay mas de un nodo fuente"
  ot = OT(dG)
  distancias[ot[0]]=0
  for i in ot:
    ltupla = dG[i]
    for tupla in ltupla:
      if(distancias[tupla[0]]>distancias[i]+tupla[1]):
        distancias[tupla[0]]=distancias[i]+tupla[1]
        padres[tupla[0]]=i
  return distancias, padres

'''
#############
# EJECUCION #
#############

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

'''
dG=TGF2dG("pg8a.txt")
print dG
print kruskal(dG)
print drBP(dG, 2)
print distMinSingleSourceDAG(dG)
print incAdy( fromAdjM2Dict( randMatrUndPosWGraph(6) ) )
print drBP ( fromAdjM2Dict( randMatrUndPosWGraph(6) ) )

fitPlotKruskal(timeKruskal(25, 10, 100, 10, 0.8, True), timeKruskal02(25, 10, 100, 10, 0.8, True), elognvalues, 10, 100, 10, 0.8)

plt.plot(timeKruskal(20, 10, 100, 10, 0.8, True),label='Kruskal compresion')
plt.plot(timeKruskal(20, 10, 100, 10, 0.8, False),label='Kruskal no compresion')
plt.legend(loc=4)
plt.ylabel("Tiempo")
plt.xlabel("Vertices")
plt.title("Kruskal con compresion VS kruskal sin compresion")
plt.show()

plt.plot(F2elognMas2nMasevalues(10, 100, 10, 0.5),label='2ElogN+2N+E')
plt.plot(F2eMas1lognMas2nvalues(10, 100, 10, 0.5),label='(2E+1)logN+2N')
plt.legend(loc=4)
plt.ylabel("Tiempo")
plt.xlabel("Vertices")
plt.title("2ElogN+2N+E VS (2E+1)logN+2N+E")
plt.show()
'''