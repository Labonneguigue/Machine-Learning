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

PrintEnabled = 0

if (len(sys.argv) > 2):
    PrintEnabled = 1

X = genfromtxt(sys.argv[1], delimiter=',')

# Number of Iterations
NbIterations = 10

# Number of clusters = 5 = K
NbClusters = 5
if (len(sys.argv) > 2):
    PrintEnabled = 1
    NbClusters = int(sys.argv[2])

# N is the number of input vectors
N = X.shape[0]
# d is the number of element per vector
d = X.shape[1]

###############################
##      K-MEANS ALGORITHM    ##
###############################

# Ci stores the number of the clusters to which the ith input vector belongs to
Ci = numpy.zeros(shape=(N,1))

# Centroids
Centroids = numpy.zeros(shape=(NbClusters,d))

# I want to go find the min & max of each input
# XminAndMax[0][i] being the min
# XminAndMax[1][i] being the max of ith element of each vector
XminAndMax = numpy.zeros(shape=(2,d))

# Ni is an array that will keep track of how many vectors belong to each clusters
# Needed in UpdateEachCentroids(): it is more efficient to update it during UpdateEachCi():
Ni = []
for e in range(NbClusters):
    Ni.append(0)


indexOverNbClusters = 0
while indexOverNbClusters < NbClusters :
    Centroids[indexOverNbClusters] = X[randint(0, N-1)]
    indexOverNbClusters += 1

#               K-means++ algorithm
#   K-means++ chooses better than random initial centroids by trying to place
#   them as far as possible from one another.

Centroids[0] = X[randint(0, N-1)]
C = 1
def GetNextCentroid(C):
    Dist = []
    for n in range(0,N-1):
        Dists = []
        for c in range(0,C):
            Dists.append(numpy.sum(numpy.multiply(X[n]-Centroids[c], X[n]-Centroids[c])))
            print("hop")
        print("finish")
        print(Dists)
        Dist.append(min(Dists))
    ProbabilityDist = Dist/sum(Dist)
    CumulativeProbability = ProbabilityDist.cumsum()
    MyRandom = random.random()
    index = 0
    result = 0
    while index < len(CumulativeProbability):
        if MyRandom < CumulativeProbability[index]:
            result = index
            break
        index += 1
    return result

indexOverNbClusters = 1
while indexOverNbClusters < NbClusters:
    Centroids[indexOverNbClusters] = X[GetNextCentroid(indexOverNbClusters)]
    indexOverNbClusters += 1

if PrintEnabled :
    print("Random generation of the initial Centroids vector :")
    print(Centroids)
    print("")

def ZeroTheArray(arr):
    for e in range(len(arr)):
        arr[e] = 0

def WriteToFile(nameOfTheFile_Prefix, nameOfTheFile_extension, OutputMatrix):
    nameOfTheFile = str(nameOfTheFile_Prefix) + str(nameOfTheFile_extension) + ".csv"
    with open(nameOfTheFile, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for e in range(OutputMatrix.shape[0]):
            spamwriter.writerow(OutputMatrix[e])
            e += 1

def UpdateEachCi():
    ZeroTheArray(Ni)
    # First while to iterate over every input vector
    indexOverN = 0
    while indexOverN < N :
        # This array stores the distance to each clusters
        Distances = []
        # Second while to calculate the euclidian distance from a vector to every clusters
        indexOverNbClusters = 0
        while indexOverNbClusters < NbClusters :
            Sum = 0
            #Third while to iterate the calcul of the distance to be the sum of the distance from all element of the input vector
            indexOverD = 0
            while indexOverD < d:
                Sum += (X[indexOverN][indexOverD]-Centroids[indexOverNbClusters][indexOverD])**2
                indexOverD += 1
            Distances.append(Sum)
            indexOverNbClusters += 1

        # We now have an array with the distance from that vector to each clusters
        assert(len(Distances) == NbClusters)
        # The number of the cluster is the index of the smallest value (array kept 0-based on purpose)
        cluster = Distances.index(min(Distances))
        Ci[indexOverN] = cluster
        Ni[cluster] += 1
        indexOverN += 1

    print("Ni : ")
    print(Ni)
    # print("Ci : ")
    # print(Ci)

def UpdateEachCentroids():
    # The number of vector that belong to each cluster is already calculated and stored in Ni
    # Reset the matrix Centroids
    global Centroids
    Centroids = numpy.zeros(shape=(NbClusters,d))

    indexOverN = 0
    while indexOverN < N :
        cluster = int(Ci[indexOverN])
        Centroids[cluster] += X[indexOverN]/Ni[cluster]
        indexOverN += 1



###############################
##      EM ALGORITHM      ##
###############################
# Mixture weight
Phi = numpy.zeros(shape=(N, NbClusters))

# Pi    Pi[k] is the expected number of points coming from the k cluster at the
#       given iteration divided by the number of points in total.
Pi = numpy.zeros(shape=(NbClusters, 1))

# We take the same initial Centroids
Centroids_EM_GMM = Centroids

def UpdateEachPhiAndPi():
    global Phi
    Phi = numpy.zeros(shape=(N, NbClusters))
    # First while to iterate over every input vector
    indexOverN = 0
    while indexOverN < N :
        # I keep track of the overall sum of the distances to normalise afterwards.
        # For a particular vector, the sum of all Phi-k = 1
        GlobalSum = 0
        # Second while to calculate the euclidian distance from a vector to every clusters
        indexOverNbClusters = 0
        while indexOverNbClusters < NbClusters :
            Sum = 0
            #Third while to iterate the calcul of the distance to be the sum of the distance from all element of the input vector
            indexOverD = 0
            while indexOverD < d:
                Sum += (X[indexOverN][indexOverD]-Centroids[indexOverNbClusters][indexOverD])**2
                indexOverD += 1
            GlobalSum += Sum
            Phi[indexOverN][indexOverNbClusters] = Sum
            indexOverNbClusters += 1

        # We now have an array with the distance from that vector to each clusters
        assert(Phi.shape[1] == NbClusters)

        # I normalize it to Sum = 1
        indexOverNbClusters = 0
        while indexOverNbClusters < NbClusters :
            Phi[indexOverN][indexOverNbClusters] /= GlobalSum

            # I update Pi right away
            # Pi-k is the sum of all the Phi-k divided by the number of inputs
            Pi[indexOverNbClusters] += Phi[indexOverN][indexOverNbClusters]/N

            indexOverNbClusters += 1

        indexOverN += 1


def UpdateEachMuAndSigma(indexOverNbIterations):
    # The number of vector that belong to each cluster is already calculated and stored in Ni
    # Reset the matrix Centroids
    global Centroids_EM_GMM
    Centroids_EM_GMM = numpy.zeros(shape=(NbClusters,d))

    indexOverN = 0
    while indexOverN < N :
        indexOverNbClusters = 0
        while indexOverNbClusters < NbClusters:
            # print("Centroids_EM_GMM[indexOverNbClusters]")
            # print(Centroids_EM_GMM[indexOverNbClusters].shape)
            # print("X[indexOverN]")
            # print(X[indexOverN].shape)
            # print("Phi[indexOverNbClusters]")
            # print(Phi[indexOverNbClusters].shape)
            Centroids_EM_GMM[indexOverNbClusters] += (X[indexOverN]*Phi[indexOverN][indexOverNbClusters])/(Pi[indexOverNbClusters]*N)
            indexOverNbClusters += 1
        indexOverN += 1

    indexOverNbClusters = 0
    while indexOverNbClusters < NbClusters:
        # New matrix Sigma for each k
        Sigma = numpy.zeros(shape=(d, d))
        indexOverN = 0
        while indexOverN < N :

            # Phi[k] * ( x[i] - Centroids_EM_GMM[k]) * transpose ( x[i] - Centroids_EM_GMM[k]) / (Pi[k] * N)
            # print("X[i].shape")
            # print(X[indexOverN].shape)
            # [numpy.newaxis] allow me to convert a 1D array into a 2D array. From there only I can transpose it
            XtoCentroid = X[indexOverN]-Centroids_EM_GMM[indexOverNbClusters]
            XtoCentroid = XtoCentroid[numpy.newaxis]
            TransposeXtoCentroid = XtoCentroid.T

            Sigma = (TransposeXtoCentroid).dot(XtoCentroid)
            Sigma *= Phi[indexOverN][indexOverNbClusters]
            Sigma /= Pi[indexOverNbClusters] * N

            indexOverN += 1

        #print(Sigma)

        WriteToFile("Sigma-"+str(indexOverNbClusters+1)+"-", indexOverNbIterations+1, Sigma)

        indexOverNbClusters += 1




###############################
##          MAIN             ##
###############################

indexOverNbIterations = 0
while indexOverNbIterations < NbIterations:

###############################
##      K-MEANS ALGORITHM    ##
###############################
    # Expectation Step
    UpdateEachCi()
    # Maximization Step
    UpdateEachCentroids()
    WriteToFile("centroids-", indexOverNbIterations+1, Centroids)


###############################
##      EM ALGORITHM         ##
###############################
    # Expectation Step
    UpdateEachPhiAndPi()
    # Maximization Step
    UpdateEachMuAndSigma(indexOverNbIterations)
    WriteToFile("pi-", indexOverNbIterations+1, Pi)
    WriteToFile("mu-", indexOverNbIterations+1, Centroids_EM_GMM)

    indexOverNbIterations += 1

color=['red','green','blue', 'yellow', 'brown']
fig=plt.figure()
ax3D = Axes3D(fig)
for e in range(0,N):
    ax3D.scatter(X[e][0], X[e][1], X[e][2], color=color[int(Ci[e])])
plt.show()
