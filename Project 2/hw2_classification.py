import csv
import sys
import numpy
import math
from numpy import genfromtxt
from numpy.linalg import inv

X_train_CSV = sys.argv[1]
y_train_CSV = sys.argv[2]
X_test_CSV = sys.argv[3]

X_train = genfromtxt(X_train_CSV, delimiter=',')
y_train = genfromtxt(y_train_CSV, delimiter=',')
X_test = genfromtxt(X_test_CSV, delimiter=',')

K = 10

i = X_train.shape[1]
n = y_train.shape[0]

Kout = X_test.shape[0]
OutputArray = numpy.zeros(shape=(Kout, K))

Pi = []
for e in range(K):
    Pi.append(0.0)

Mu = numpy.zeros(shape=(K,i))
Sigma = numpy.zeros(shape=(i*K,i))

e = 0
while e < y_train.shape[0]:
    Pi[int(y_train[e])]     += 1.0
    Mu[int(y_train[e])]     += X_train[e]
    e += 1

for e in range(K):
    Pi[e] /= float(y_train.shape[0])
    Mu[e] /= (float(y_train.shape[0])*Pi[e])

print("Pi : ")
print(Pi)

print("Mu : ")
print(Mu)

print("Sigma : ")
print(Sigma)

e = 0
while e < n:
    xi = X_train[e]
    y  = int(y_train[e])
    verticale = numpy.zeros(shape=(i,1))

    i_locale = 0
    while i_locale < i :
        verticale[i_locale]=xi[i_locale]-Mu[y][i_locale]
        i_locale += 1

    horizontale = numpy.transpose(verticale)
    temp = verticale.dot(horizontale)

    i_locale = 0
    while i_locale < i :
        Sigma[i*y+i_locale]  += temp[i_locale]
        i_locale += 1

    e += 1

index = 0
classe = 0
while index < Sigma.shape[0]:
    if index == i:
        classe += 1
    Sigma[index] /= (float(y_train.shape[0])*Pi[classe])
    index += 1

print("Sigma : ")
print(Sigma)

e = 0
while e < Kout :
    f = 0
    Sum = 0
    while f < K :
        SigmaLocale = numpy.zeros(shape=(i,i))
        i_locale = 0
        while i_locale < i:
            SigmaLocale[i_locale] = Sigma[i*f+i_locale]
            i_locale += 1

        OutputArray[e][f] = Pi[f]*numpy.linalg.det(SigmaLocale)**(-1/2)
        verticale = numpy.zeros(shape=(i,1))

        i_locale_2 = 0
        while i_locale_2 < i :
            verticale[i_locale_2]=X_test[e][i_locale_2]-Mu[f][i_locale_2]
            i_locale_2 += 1

        horizontale = numpy.transpose(verticale)

        OutputArray[e][f] *= math.exp(-0.5*(horizontale.dot(inv(SigmaLocale)).dot(verticale)))
        Sum += OutputArray[e][f]
        f += 1

    f = 0
    while f < K:
        OutputArray[e][f] *= 1/Sum
        f += 1

    e += 1

print("OutputArray")
print(OutputArray)

nameOfTheFile = "probs_test.csv"
print(nameOfTheFile)

with open(nameOfTheFile, 'wb') as csvfile:
    spamwriter2 = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    e = 0
    while e < Kout:
        spamwriter2.writerow(OutputArray[e])
        e += 1
