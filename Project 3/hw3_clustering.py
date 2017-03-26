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
#import time
#start_time = time.time()


PrintEnabled = 0

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
    print("K-means++ Centroids Initialization:")
    print(Centroids)
    print("")

def ZeroTheArray(arr):
    for e in range(len(arr)):
        arr[e] = 0

def InitAnArrayOfSize(size):
    arr = []
    while size > 0:
        arr.append(0.0)
        size -= 1
    return arr

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
        index = 0
        while index < NbClusters:
            print(OutputArray[index])
            spamwriter.writerow(OutputArray[index])
            index += 1

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


printOnlyOnce = 0
def UpdateEachPhiAndPi():
    global Pi
    global Phi
    global printOnlyOnce
    Phi = numpy.zeros(shape=(N, NbClusters))

    # First while to iterate over every input vector

    # E-step    --      Phi
    indexOverN = 0
    while indexOverN < N :
        indexOverNbClusters = 0
        SumPhiForThisCluster = 0.0
        while indexOverNbClusters < NbClusters :
            SigmaK = numpy.zeros(shape=(d, d))
            SigmaK = MatrixOfSigmas[indexOverNbClusters*d:(indexOverNbClusters+1)*d]

            Determinant = numpy.linalg.det(SigmaK)
            XminusMu = X[indexOverN] - Centroids_EM_GMM[indexOverNbClusters]

            TransposeXminusMu = numpy.transpose(XminusMu[numpy.newaxis])

            SigmaInverse = inv(SigmaK)
            PiDet = Pi[0][indexOverNbClusters] * Determinant**(-0.5)

            MatrixMul = XminusMu[numpy.newaxis].dot(SigmaInverse).dot(TransposeXminusMu)

            Exp = math.exp(-0.5 * MatrixMul)
            if printOnlyOnce == 0 :
                print("SigmaK")
                print(SigmaK)
                print("XminusMu")
                print(XminusMu)
                print("TransposeXminusMu")
                print(TransposeXminusMu)
                print("SigmaInverse")
                print(SigmaInverse)
                print("Pi[0][indexOverNbClusters]")
                print(Pi[0][indexOverNbClusters])
                print("PiDet")
                print(PiDet)
                print("Determinant")
                print(Determinant)
                print("MatrixMul")
                print(MatrixMul)
                print("Exp")
                print(Exp)
                printOnlyOnce += 1
            # assert(Exp > 0)
            Phi[indexOverN][indexOverNbClusters] = PiDet * Exp
            # assert(Phi[indexOverN][indexOverNbClusters] > 0)
            #Pi[0][indexOverNbClusters] += Phi[indexOverN][indexOverNbClusters]
            SumPhiForThisCluster += Phi[indexOverN][indexOverNbClusters]
            #print("Phi of N = " + str(indexOverN) + " for cluster = " + str(indexOverNbClusters))
            #print(Phi[indexOverN][indexOverNbClusters])
            indexOverNbClusters += 1

        # Divide by the sum of the Pi * MultivariateNormal for each K
        # print("Phi[indexOverN] before sum = 1")
        # print(Phi[indexOverN])
        # print("Pi")
        # print(Pi)
        # print(sum(Pi[0]))
        Phi[indexOverN] = Phi[indexOverN]/SumPhiForThisCluster
        Pi[0] += Phi[indexOverN]
        # print("Phi[indexOverN] after sum = 1")
        # print(Phi[indexOverN])
        # print("Sum of phi")
        # print(sum(Phi[indexOverN]))
        indexOverN += 1

    #  Pi[k]
    # indexOverN = 0
    # while indexOverN < N :
    #     indexOverNbClusters = 0
    #     while indexOverNbClusters < NbClusters :
    #         # Pi-k is the sum of all the Phi-k divided by the number of inputs
    #         Pi[0][indexOverNbClusters] += Phi[indexOverN][indexOverNbClusters]/N
    #         indexOverNbClusters += 1
    #     indexOverN += 1
    Pi[0] /= float(N)


def UpdateEachMuAndSigma(indexOverNbIterations):
    # The number of vector that belong to each cluster is already calculated and stored in Ni
    # Reset the matrix Centroids
    global Centroids_EM_GMM
    global MatrixOfSigmas
    Centroids_EM_GMM = numpy.zeros(shape=(NbClusters,d))

    # print("Pi")
    # print(Pi)
    # print("MatrixOfSigmas")
    # print(MatrixOfSigmas)

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
            Centroids_EM_GMM[indexOverNbClusters] += (X[indexOverN]*Phi[indexOverN][indexOverNbClusters])/(Pi[0][indexOverNbClusters]*N)
            indexOverNbClusters += 1
        indexOverN += 1

    # print("Centroids_EM_GMM")
    # print(Centroids_EM_GMM)

    indexOverNbClusters = 0
    while indexOverNbClusters < NbClusters:
        # New matrix Sigma for each k
        Sigma = numpy.zeros(shape=(d, d))
        indexOverN = 0
        while indexOverN < N :
            SigmaN = numpy.zeros(shape=(d, d))
            # Phi[k] * ( x[i] - Centroids_EM_GMM[k]) * transpose ( x[i] - Centroids_EM_GMM[k]) / (Pi[0][k] * N)
            # print("X[i].shape")
            # print(X[indexOverN].shape)
            # [numpy.newaxis] allow me to convert a 1D array into a 2D array. From there only I can transpose it
            XtoCentroid = X[indexOverN]-Centroids_EM_GMM[indexOverNbClusters]
            XtoCentroid = XtoCentroid[numpy.newaxis]
            TransposeXtoCentroid = XtoCentroid.T

            SigmaN = (TransposeXtoCentroid).dot(XtoCentroid)
            SigmaN *= Phi[indexOverN][indexOverNbClusters]
            Sigma += SigmaN
            indexOverN += 1

        #print(Sigma)
        Sigma /= (Pi[0][indexOverNbClusters] * N)
        MatrixOfSigmas[indexOverNbClusters*d:(indexOverNbClusters+1)*d] = Sigma

        WriteToFile("Sigma-"+str(indexOverNbClusters+1)+"-", indexOverNbIterations+1, Sigma)

        indexOverNbClusters += 1

    # print("MatrixOfSigmas")
    # print(MatrixOfSigmas)


def InitSigmaToIdentityMatrix():
    global MatrixOfSigmas
    k = 0
    indexOverN = 0
    while indexOverN < N:
        #   I retrieve the cluster the data point belongs to
        y = Ci[indexOverN][0]
        XtoCentroid = X[indexOverN] - Centroids[k]
        XtoCentroid = XtoCentroid[numpy.newaxis]
        TransposeXtoCentroid = XtoCentroid.T
        MatrixOfSigmas[y*d:(y+1)*d] += (XtoCentroid).dot(TransposeXtoCentroid)
        indexOverN += 1

    indexOverNbClusters = 0
    while indexOverNbClusters < NbClusters:
        SigmaK = MatrixOfSigmas[indexOverNbClusters*d:(indexOverNbClusters+1)*d]
        SigmaK /= Ni[indexOverNbClusters]
        MatrixOfSigmas[indexOverNbClusters*d:(indexOverNbClusters+1)*d] = SigmaK
        indexOverNbClusters += 1

    indexOverDCluster = 0
    indexOverd = 0
    while indexOverDCluster < d*NbClusters:
        MatrixOfSigmas[indexOverDCluster][indexOverd] += 1
        indexOverd += 1
        if indexOverd >= d:
            indexOverd = 0
        indexOverDCluster += 1

    print(MatrixOfSigmas)


###############################
##          MAIN             ##
###############################




###############################
##      K-MEANS ALGORITHM    ##
###############################

indexOverNbIterations = 0
while indexOverNbIterations < NbIterations:
    # Expectation Step
    UpdateEachCi()
    # Maximization Step
    UpdateEachCentroids()
    WriteToFile("centroids-", indexOverNbIterations+1, Centroids)
    indexOverNbIterations += 1




###############################
##      GMM ALGORITHM        ##
###############################

#       Probability of each data point to belong to each cluster
Phi = numpy.zeros(shape=(N, NbClusters))

#       Pi[0][k] is a K-dimensional probability distribution.
#       Basically the weight of each Gaussian.
#       Initialized to be the uniform distribution
Pi = numpy.zeros(shape=(1,NbClusters))

indexOverNbClusters = 0
while indexOverNbClusters < NbClusters:
    Pi[0][indexOverNbClusters] = (1/float(NbClusters))
    indexOverNbClusters += 1
print("Pi")
print(Pi)

#       Stores verticaly every Sigma
MatrixOfSigmas = numpy.zeros(shape=(d*NbClusters, d))
InitSigmaToIdentityMatrix()

#       Initialization of the centroids to be the result of the K-means algo
Centroids_EM_GMM = Centroids


indexOverNbIterations = 0
while indexOverNbIterations < NbIterations:
    # Expectation Step
    UpdateEachPhiAndPi()
    # Maximization Step
    UpdateEachMuAndSigma(indexOverNbIterations)
    # PiMatrix = numpy.mat(Pi)
    # print(PiMatrix)
    # PiMatrix = numpy.transpose(PiMatrix[numpy.newaxis])
    # print(PiMatrix)
    WriteToFile("pi-", indexOverNbIterations+1, numpy.transpose(Pi))
    WriteToFile("mu-", indexOverNbIterations+1, Centroids_EM_GMM)
    indexOverNbIterations += 1

print("")
print("Finishes OK.")
print("")

# print("--- %s seconds ---" % (time.time() - start_time))


# Print a 3D graph.
# Each color represents the belonging to a certain cluster.

if PrintEnabled:
    color=['red','green','blue', 'yellow', 'brown']
    fig=plt.figure()
    ax3D = Axes3D(fig)
    for e in range(0,N):
        ax3D.scatter(X[e][0], X[e][1], X[e][2], color=color[int(Ci[e])])
    plt.show()
