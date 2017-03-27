import csv
import sys
import numpy
from numpy import genfromtxt
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

Lambda = float(sys.argv[1])
Sigma2 = float(sys.argv[2])

X_train = genfromtxt(sys.argv[3], delimiter=',')
y_train = genfromtxt(sys.argv[4], delimiter=',')
X_test = genfromtxt(sys.argv[5], delimiter=',')

# Get the number of columns -> dimension of the input vector -> size of identity_matrix
N = X_train.shape[0]
d = X_train.shape[1]
N_test = X_test.shape[0]

######################
##      PART 1      ##
######################

identity_matrix = numpy.identity(d)
LambdaDotIdentityMatrix = numpy.multiply(identity_matrix,Lambda)
XTransposeX = numpy.transpose(X_train).dot(X_train)
Inverse = inv(LambdaDotIdentityMatrix+XTransposeX)
XtransposeY = numpy.transpose(X_train).dot(y_train)
wRR = Inverse.dot(XtransposeY)

nameOfThePart1File = "wRR_"+str(int(Lambda))+".csv"
with open(nameOfThePart1File, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(numpy.transpose(wRR))

if Sigma2 > 2 :
    color=['brown', 'blue']
    fig=plt.figure()
    ax3D = Axes3D(fig)

    point1  = numpy.array([0,0,wRR[2]])
    normal1 = numpy.array([wRR[0],wRR[1],-1])
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d1 = -numpy.sum(point1*normal1)# dot product
    # create x,y
    xx, yy = numpy.meshgrid(range(2), range(2))
    # calculate corresponding z
    z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]
    # plot the surface
    # plt3d = plt.figure().gca(projection='3d')
    ax3D.plot_surface(xx,yy,z1, color='yellow', antialiased=True)

    ax3D.set_xlabel("x1 : Dimension 1 input")
    ax3D.set_ylabel("x2 : Dimension 2 input")
    ax3D.set_zlabel("y1 : Output")
    for e in range(0,N):        # Plot training data
        ax3D.scatter(X_train[e][0], X_train[e][1], y_train[e], color=color[1])
    for e in range(0,N_test):        # Plot test data
        ax3D.scatter(X_test[e][0], X_test[e][1], wRR.dot(X_test[e]), color=color[0])

    plt.show()


#############
#   Part2   #
#############


def SetSigmaOs(Sigma, Test_Set_Local):
    Sigma0Array_Local = numpy.arange(float(Test_Set_Local.shape[0]))
    index = 0
    for row in Test_Set_Local:
        Sigma0Array_Local[index] = float(Sigma2 + numpy.transpose(row).dot(Sigma).dot(row))
        index += 1
    return Sigma0Array_Local

def CalculPosterior(SigmaPrior_Local,row, Array_Test):
    SigmaPosterior = numpy.multiply(1/Sigma2, Array_Test[row,:])
    SigmaPosterior = numpy.multiply(SigmaPosterior, numpy.transpose(Array_Test[row,:]))
    SigmaPosterior = SigmaPosterior + inv(SigmaPrior_Local)
    SigmaPosterior = inv(SigmaPosterior)
    return SigmaPosterior

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def ReturnHighestValuesIndex(value, array):
    index=0
    while index<array.shape[0]:
        if isclose(array[index],value):
            return index
        index += 1

def GetCorrectiveValue(Local_Highest,indexGlobal_Local):
    print("Hop")
    print(Local_Highest)
    local_index = 0
    Local_Highest += indexGlobal_Local
    while local_index<indexGlobal_Local:
        print("local_index" + str(local_index))
        if OutputArray[local_index]>Local_Highest:
            print("in")
            print(OutputArray[local_index])
            Local_Highest -= 1
        local_index += 1
    return Local_Highest

def CalcEntropy(Entropy_arr, x0):
    Hprior = Entropy_arr[-1]
    ln = numpy.log(1 + (1/Sigma2)*numpy.transpose(Test_Set[x0,:]).dot(SigmaPrior).dot(Test_Set[x0,:]))
    Hpost = Hprior - ln * d / 2
    Entropy_arr.append(Hpost)
    return Entropy_arr

SigmaPriorInverse = LambdaDotIdentityMatrix + numpy.multiply(1/Sigma2, XTransposeX)
SigmaPrior = inv(SigmaPriorInverse)
Test_Set = X_test
OutputArray = []
Entropy = []
#Arbitrary prior entropy
Entropy.append(0.0)

indexGlobal = 0
while indexGlobal < 50:
    # Part2
    Sigma0Array = SetSigmaOs(SigmaPrior, Test_Set);

    # Part3
    Sigma0ArraySorted = numpy.sort(Sigma0Array,axis=0)
    HighestValueIndex = ReturnHighestValuesIndex(Sigma0ArraySorted[-1], Sigma0Array)
    SavedHighestValueIndex = HighestValueIndex
    Entropy = CalcEntropy(Entropy, HighestValueIndex)
    print(HighestValueIndex)
    #Correction of the index as we iteratively delete the chosen data point
    HighestValueIndex += indexGlobal
    ToBeSubstracted = 0
    for e in OutputArray:
        if e > HighestValueIndex:
            ToBeSubstracted += 1
    HighestValueIndex = HighestValueIndex - ToBeSubstracted
    OutputArray.append(HighestValueIndex)

    # Part4
    Test_Set = numpy.delete(Test_Set, (SavedHighestValueIndex), axis=0)

    # Part5
    SigmaPosterior = CalculPosterior(SigmaPrior,HighestValueIndex, X_train)
    SigmaPrior = SigmaPosterior
    indexGlobal += 1

nameOfThePart2File = "active_"+str(int(Lambda))+"_"+str(int(Sigma2))+".csv"
print(nameOfThePart2File)

with open(nameOfThePart2File, 'wb') as csvfile2:
    spamwriter2 = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter2.writerow(OutputArray)

print (Entropy)
plt.plot(Entropy)
plt.ylabel('Entropy')
plt.show()
# index= X_test.shape[0]-1
# while index>X_test.shape[0]-11 :
#     index2 = 0
#     # maxi = 0
#     # IndexMaxi = 0
#     while index2 < Sigma0Array.shape[0]:
#         if Sigma0Array[index2] == Sigma0ArraySorted[index]:
#             #print(index2)
#             CalculPosterior(index2)
#         index2 += 1
#     # numpy.delete(Sigma0Array, IndexMaxi)
#     # AlreadyOut[index] = IndexMaxi
#     index -= 1
# #




# #print(numpy.transpose(X_train))

# f = open ( 'input.txt' , 'r')
# l = [ map(int,line.split(',')) for line in f ]
# #print l

# with fileinput.input(files=('X_train.csv')) as f:
#     for line in f:
#         process(line)


#python hw1_regression.py lambda sigma2  y_train.csv X_test.csv

#Page total likes;Type;Category;Post Month;Post Weekday;Post Hour;Paid;Lifetime Post Total Reach;Lifetime Post Total Impressions;Lifetime Engaged Users;Lifetime Post Consumers;Lifetime Post Consumptions;Lifetime Post Impressions by people who have liked your Page;Lifetime Post reach by people who like your Page;Lifetime People who have liked your Page and engaged with your post;comment;like;share;Total Interactions
