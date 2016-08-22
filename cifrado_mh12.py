# -*- encoding: utf-8 -*-

#Imports
import sys
import random
import numpy as np
import copy

##############
##############
# PRACTICA 3 #
##############
##############

##################################
# PARTE 1 - Cifrado Merkle–Hellman
##################################

## Sucesiones supercrecientes

def genSuperCrec(n_terms):
  """
  Genera una sucesion super creciente
  :param n_terms: numero de elementos de la sucesión super creciente
  :return: array con los numero de la sucesion super creciente
  """
  res = []
  res.append(np.random.randint(10)+1)
  for i in range(n_terms-1):
    sum=0
    for j in range(i+1):
      sum=sum+res[j-1]
    new=random.randrange(sum+1, sum+10, 1)
    res.append(new)
  return res

## Modulo, multiplicador e inverso, y sucesion publica

def multiplier(mod, mult_ini):
  """
  Devuelve un entero primo relativo con modulo mod superior a mult_ini
  :param mod: modulo
  :param mult_ini: resultado minimo admitido
  :return: primo relativo con mod
  """
  while 1:
    b = random.randrange(mult_ini, mod, 1)
    if mcd(mod, b)==1:
  	  return b

def mcd(x, y):
  """
  Resuelve el mcd de dos numeros
  :param x: numero 1
  :param y: numero 2
  :return: mcd de x e y
  """
  while y != 0:
    x, y = y, x % y
  return x

def inverse(p, mod):
  """
  Calcula el inverso de p%mod
  :param p: numero
  :param mod: modulo
  :return: inverso de p%mod
  """
  for i in range(mod):
    if i*p%mod == 1:
      return i
  return 

def inverse2(p, mod):
  """
  Calcula el inverso de p%mod
  :param p: numero
  :param mod: modulo
  :return: inverso de p%mod
  """
  a, x, y = egcd1(p, mod)
  if a!=1:
  	return
  return x%mod

def inverse3(p, mod):
  """
  Calcula el inverso de p%mod
  :param p: numero
  :param mod: modulo
  :return: inverso de p%mod
  """
  a, x, y = egcd2(p, mod)
  if a!=1:
  	return
  return x%mod

def egcd1(a, b):
  """
  Algoritmo de Euclides Extendido (egcd)
  :param a: numero 1
  :param b: numero 2
  :return: parametros b, x, y del egcd
  """
  x, y, u, v = 0, 1, 1, 0
  while a != 0:
    q, r = b//a, b%a
    m, n = x-u*q, y-v*q
    b,a, x,y, u,v = a,r, u,v, m,n
  return b, x, y

def egcd2(a, b):
  """
  Algoritmo de Euclides Extendido (egcd)
  :param a: numero 1
  :param b: numero 2
  :return: parametros b, x, y del egcd
  """
  if a == 0:
    return (b, 0, 1)
  else:
    g, y, x = egcd2(b % a, a)
    return (g, x - (b // a) * y, y)

def modMultInv(lSC):
  """
  Calcula un modulo para una suceesion super creciente
  :param lSC: sucesion super creciente
  :return: multiplicador, inverso del multiplicador, modulo para la suceesion super creciente
  """
  sum=0
  for i in range(len(lSC)):
    sum=sum+lSC[i]
  mod=random.randrange(sum+1, sum+50, 1) 
  p=multiplier(mod, 50) 
  q=inverse2(p, mod)
  pub=genSucesionPublica(lSC, p, mod)
  sum=0
  for i in range(len(pub)):
  	sum=sum+pub[i]
  if sum*q > sys.maxint:
  	return modMultInv(lSC)
  else:
  	return p, q, mod

def genSucesionPublica(lSC, p, mod):
  """
  Genera una sucesion publica
  :param lSC: sucesion super creciente privada
  :param p: multiplicador
  :param mod: modulo
  :return: sucesion publica
  """
  res=[]
  for i in range(len(lSC)):
    res.append((lSC[i]*p)%mod)
  return res

def lPub_2_lSC(l_pub, q, mod):
  """
  Genera la sucesion privada
  :param l_pub: sucesion publica
  :param q: inverso
  :param mod: modulo
  :return: sucesion privada
  """
  res=[]
  for i in range(len(l_pub)):
     res.append((l_pub[i]*q)%mod)
  return sorted(res) 	

##Cifrado de cadenas binarias y su descifrado

def genRandomBitString(n_bits):
  """
  Devuelve una lista de bits aleatorios
  :param n_bits: tamaño de la lista
  :return: lista de bits
  """
  res=[]
  for i in range(n_bits):
    res.append(random.randrange(0, 2, 1))
  return res

def MH_encrypt(s, lPub, mod):
  """
  Encripta con el metodo Merkle-Hellman
  :param s: lista binaria
  :param lPub: sucesion publica
  :param mod: modulo
  :return: mensaje cifrado
  """
  res=[]
  while len(s)%len(lPub)!=0:
  	s.append(0) 
  num_sec=len(s)/len(lPub)
  s1=np.array(s).reshape(num_sec, len(lPub))
  for fila in range(num_sec):
    sum=0
    for columna in range(len(lPub)):
      sum=sum+lPub[columna]*s1[fila, columna]
    res.append(sum)
  return res

def block_decrypt(C, l_sc, inv, mod):
  """
  Descifra un bloque cifrado con el metodo Merkle-Hellman
  :param C: entero cifrado
  :param l_sc: sucesion privada
  :param inv: inverso
  :param mod: modulo
  :return: bloque descifrado
  """
  res=[]
  aux=C*inv%mod
  for elemento in reversed(l_sc):
    if aux >= elemento:
      aux-=elemento
      res.append(1)
    else:
      res.append(0)
  return res[::-1]

def l_decrypt(l_cifra, l_sc, inv, mod):
  """
  Descifra un mensaje cifrado con el metodo Merkle-Hellman
  :param l_cifra: mensaje cifrado
  :param l_sc: sucesion privada
  :param inv: inverso
  :param mod: modulo
  :return: mensaje descifrado
  """
  res=[]
  for i in range(len(l_cifra)):
    res.extend(block_decrypt(l_cifra[i], l_sc, inv, mod))
  return res

#############
# EJECUCION #
#############
'''
def main(argv):
  return

if __name__ == "__main__":
  main(sys.argv)
'''
'''
print mcd(100, 15)
print multiplier(80, 40)
print inverse(11, 36)
print inverse2(7, 36)
print inverse3(7, 36)

a = genSuperCrec(6)
print a
p, q, mod = modMultInv(a)
print p, q, mod
publica = genSucesionPublica(a, p, mod)
print publica
privada = lPub_2_lSC(publica, q, mod)
print privada
bits = genRandomBitString(6)
print "-->", bits
cosa = MH_encrypt(bits, publica, mod)
print cosa
print "-->", l_decrypt(cosa, privada, q, mod)

suma=0
suma2=0
for i in range(30):
  suma+=2**i
  suma2+=suma2+suma
print suma
print suma2

acc=0
tot=0
for i in range(12):
  tot+=acc
  acc=tot+10
print acc
'''
