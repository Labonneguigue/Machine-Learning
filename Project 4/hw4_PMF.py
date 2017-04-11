import csv
import sys
import numpy
import math
from numpy import genfromtxt
from numpy.linalg import inv
import random
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
start_time = time.time()


#   Given values
d = 5               #   number of dimensions to learn / degrees of freedom
Sigma2 = 0.1
Lambda = 2
NbIterations = 50

# Ratings.csv is structured : userId,movieId,rating,timestamp
Ratings = genfromtxt(sys.argv[1], delimiter=',')

N = Ratings.shape[0]

N1 = 0  #Nb of users
N2 = 0  #Nb of objects

#   I find the number of Users (N1) and the number of objects rated (N2)
indexOverN = 0
while indexOverN < N:
    if Ratings[indexOverN][0] > N1:
        N1 = int(Ratings[indexOverN][0])
    if Ratings[indexOverN][1] > N2:
        N2 = int(Ratings[indexOverN][1])
    indexOverN += 1

#   Withdraw the unused objects
#
# Count = []
# index = 0
# while index < N2+1:
#     Count.append(0)
#     index += 1
#
# indexOverN = 0
# while indexOverN < N:
#     Count[int(Ratings[indexOverN][1])] += 1
#     indexOverN += 1
#
# counter = 0
# j = 1
# while j < N2+1:
#     save = int(Count[j])
#     Count[j] = int(counter)
#     if save == 0:
#         counter += 1
#     j += 1
#
# N2 = 0
# indexOverN = 0
# while indexOverN < N:
#     Ratings[indexOverN][1] = Ratings[indexOverN][1] - Count[int(Ratings[indexOverN][1])]
#     if Ratings[indexOverN][1] > N2:
#         N2 = Ratings[indexOverN][1]
#     indexOverN += 1
#
# with open("corrected-ratings-sandipan.csv", 'wb') as csvfile:
#     spamwriter2 = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     e = 0
#     while e < N:
#         spamwriter2.writerow(Ratings[e])
#         e += 1

print("N1 (Nb of users)")
print(N1)
print("N2 (Nb of objects)")
print(N2)


# Populate OmegaIJ
OmegaIJ = numpy.zeros(shape=(N1, N2))
indexOverN = 0
while indexOverN < N:
    OmegaIJ[Ratings[indexOverN][0]-1][Ratings[indexOverN][1]-1] = Ratings[indexOverN][2]
    indexOverN += 1

U = numpy.zeros(shape=(N1, d))
V = numpy.zeros(shape=(d, N2))

#   I choose to Initialize V to a normal distribution 0 mean and lambda^-1 * I covariance
indexOverD = 0
while indexOverD < d:
    indexOverN2 = 0
    while indexOverN2 < N2:
        V[indexOverD][indexOverN2] = numpy.random.normal(0,float(1.0/Lambda))
        indexOverN2 += 1
    indexOverD += 1

indexOverD = 0
while indexOverD < d:
    indexOverN1 = 0
    while indexOverN1 < N1:
        U[indexOverN1][indexOverD] = numpy.random.normal(0,float(1.0/Lambda))
        indexOverN1 += 1
    indexOverD += 1


#   MAP objective function to minimize
L = []

def CalculL():
    global L
    Sum_Mij_uivj = 0
    Sum_ui = 0
    Sum_vj = 0
    i = 0
    while i < N1:
        j = 0
        while j < N2:
            if OmegaIJ[i][j] > 0:
                Ui = U[i]
                Vj_ = V[:,j]
                Vj = V[:,j][numpy.newaxis]
                UV = Ui.dot(Vj.T)
                Sum_Mij_uivj += (OmegaIJ[i][j] - UV)**2 / (2*Sigma2)
                Sum_ui += numpy.linalg.norm(Ui, ord=2) * Lambda / 2
                Sum_vj += numpy.linalg.norm(Vj_, ord=2) * Lambda / 2
            j+=1
        i += 1
    newL = - Sum_Mij_uivj - Sum_ui - Sum_vj
    L.append(int(newL))

def WriteToFile(nameOfTheFile_Prefix, nameOfTheFile_extension, OutputMatrix):
    nameOfTheFile = str(nameOfTheFile_Prefix) + str(nameOfTheFile_extension) + ".csv"
    with open(nameOfTheFile, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for e in range(OutputMatrix.shape[0]):
            spamwriter.writerow(OutputMatrix[e])

def WriteArrayToFile(nameOfTheFile_Prefix, nameOfTheFile_extension, OutputArray):
    nameOfTheFile = str(nameOfTheFile_Prefix) + str(nameOfTheFile_extension) + ".csv"
    with open(nameOfTheFile, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for obj in range(len(OutputArray)):
            spamwriter.writerow([OutputArray[obj]])


iteration = 0
while iteration < NbIterations:
    #Update User location
    i = 0
    while i < N1:

        identity_matrix = numpy.identity(d)
        LSI = identity_matrix * Lambda * Sigma2
        Sum = numpy.zeros(shape=(d,d))
        Mij_vj = numpy.zeros(shape=(d,1))

        j = 0
        while j < N2:
            if OmegaIJ[i][j] > 0:          # Meaning that this rating is present
                Vj = V[:,j][numpy.newaxis]
                Sum += (Vj.T).dot(Vj)
                # print("Vj")
                # print(Vj)
                # print("outer product")
                # print(Sum)
                Mij_vj += numpy.transpose(numpy.multiply(Vj,OmegaIJ[i][j]))
            j += 1
        # if (i == 6 or i == 7):
        #     print("Sum")
        #     print(Sum)
        #     print("Mij_vj")
        #     print(Mij_vj)
        LeftProduct = LSI + Sum
        LeftProduct = inv(LeftProduct)
        Total = LeftProduct.dot(Mij_vj)
        U[i] = Total.T
        # if (i == 6 or i == 7):
        #     print("LeftProduct")
        #     print(LeftProduct)
        #     print("Total")
        #     print(Total)
        i += 1

    # print("U")
    # print(U)

    #   update the object location
    j = 0
    while j < N2:
        identity_matrix = numpy.identity(d)
        LSI = identity_matrix * Lambda * Sigma2
        Sum = numpy.zeros(shape=(d,d))
        Mij_ui = numpy.zeros(shape=(d,1))
        i = 0
        while i < N1:
            if OmegaIJ[i][j] > 0:          # Meaning that this rating is present
                Ui = U[i][numpy.newaxis]
                Sum += numpy.transpose(Ui).dot(Ui)
                # print("yyy")
                # print(Sum)
                Mij_ui += numpy.transpose(numpy.multiply(Ui,OmegaIJ[i][j]))
            i += 1
        LeftProduct = LSI + Sum
        LeftProduct = inv(LeftProduct)
        Total = LeftProduct.dot(Mij_ui)
        V[:,j] = Total.T
        j += 1


    # print("iteration")
    # print(iteration)
    CalculL()
    # print("L")
    # print(L[-1])
    iteration += 1
    if (iteration == 10 or iteration == 25 or iteration == 50):
        WriteToFile("U-", iteration, U)
        WriteToFile("V-", iteration, V.T)

WriteArrayToFile("objective", "", L)

print("--- %s seconds ---" % (time.time() - start_time))

# plt.plot(L)
# plt.ylabel('L')
# plt.show()
