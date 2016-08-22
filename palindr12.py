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

#######################
# PARTE 1 - Palindromos
#######################

#Pilas
def initS():
  return []

def emptyS(S):
  if not S:
    return False
  return True

def push(elem, S):
  S.append(elem)

def pop(S):
  return S.pop()

#Palindromos
def isPalyndrome(S):
  AList=initS()
  for i in S:
    push(i, AList)
  for i in S:
    if i != pop(AList):
      return False
  return True

def isPalyndrome2(S):
  if S != S[::-1]:
      return False
  return True

#Cadenas aleatorias
def randomString(long, strBase):
  s=""
  for i in range(long):
    s = s + random.choice(strBase)
  return s

#Palindromos aleatorios
def randomPalyndrome(long, strBase):
  res = randomString(long//2, strBase)
  if((long%2)==0): #Palabra con caracteres pares
    res = res + res[::-1]
  else: #Con caracteres impares
    res = res + random.choice(strBase) + res[::-1]
  return res

#Lista con prob de ser palindromos
def generateRandomPalyndrList(numSt, longMax, probPalyndrome):
  res = []
  for i in range (numSt):
    if(probPalyndrome > random.randrange(101)): #Para generar entre 0 y 100
      res.append(randomPalyndrome(random.randrange(longMax)+1, string.ascii_letters))
    else:
      res.append(randomString(random.randrange(longMax)+1, string.ascii_letters))
  return res

#Contar palindromos en una lista
def countPalyndromesList(l):
  count=0
  for i in l:
    if( isPalyndrome(i) == True ):
      count=count+1
  return count

#Guardar cada lista en una linea del archivo
def list2file(l, fName):
  with open(fName, "w") as f:
    f.write("\n".join(l))
  return

#Contar palindromos en un archivo
def countPalyndromesFile(fName):
  with open(fName) as f:
    l = f.read().splitlines()
  return countPalyndromesList(l)

#Contar lineas de un archivo
def countLinesFile(fName):
  count = 0
  with open(fName) as f:
  	l = f.read().splitlines()
  return len(l)

#Ejecucion
def main(argv):
  fName = argv[1]
  numLines = countLinesFile(fName)
  numPalyndr = countPalyndromesFile(fName)
  print("fName %s:\tnum sequences: %d\tnum palyndromes: %d" % (fName, numLines, numPalyndr))
  return

if __name__ == "__main__":
  main(sys.argv)