import csv
import sys
import numpy
from numpy import genfromtxt
from numpy.linalg import inv

Lambda = float(sys.argv[1])
Sigma2 = float(sys.argv[2])

#print("Lambda = ")
#print(Lambda)
#print("Sigma2 = ")
#print(Sigma2)

X_train_CSV = sys.argv[3]
y_train_CSV = sys.argv[4]
X_test_CSV = sys.argv[5]


#X_train = genfromtxt('X_train.csv', delimiter=',')
X_train = genfromtxt(X_train_CSV, delimiter=',')
##print("X_train = ")
##print(X_train)

#y_train = genfromtxt('y_train.csv', delimiter=',')
y_train = genfromtxt(y_train_CSV, delimiter=',')
##print("y_train = ")
##print(y_train)

#X_test = genfromtxt('X_test.csv', delimiter=',')
X_test = genfromtxt(X_test_CSV, delimiter=',')
##print("X_test = ")
##print(X_test)

# Get the number of columns -> size of identity_matrix
# Shape returns a tuple (rows, columns)
columns = X_train.shape[1]

identity_matrix = numpy.identity(columns)

LambdaDotIdentityMatrix = numpy.multiply(identity_matrix,Lambda)
##print("LambdaDotIdentityMatrix = ")
##print(LambdaDotIdentityMatrix)

XTransposeX = numpy.transpose(X_train).dot(X_train)
##print("XTransposeX")
##print(XTransposeX)

Inverse = inv(LambdaDotIdentityMatrix+XTransposeX)
##print("Inverse")
##print(Inverse)

XtransposeY = numpy.transpose(X_train).dot(y_train)
# #print("XtransposeY")
# #print(XtransposeY)

wRR = Inverse.dot(XtransposeY)
#print("wRR")
#print(wRR)

nameOfThePart1File = "wRR_"+str(int(Lambda))+".csv"
#print(nameOfThePart1File)

with open(nameOfThePart1File, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(numpy.transpose(wRR))


#############
#   Part2   #
#############

def SetSigmaOs(Sigma, Test_Set_Local):
    #print("Step 2 : ")
    #print("SetSigmaOs")
    Sigma0Array_Local = numpy.arange(float(Test_Set_Local.shape[0]))
    index = 0
    for row in Test_Set_Local:
        Sigma0Array_Local[index] = float(Sigma2 + numpy.transpose(row).dot(Sigma).dot(row))
        index += 1
    return Sigma0Array_Local

def CalculPosterior(SigmaPrior_Local,row, Array_Test):
    #print("Step 5 : ")
    #print("CalculPosterior")
    SigmaPosterior = numpy.multiply(1/Sigma2, Array_Test[row,:])
    SigmaPosterior = numpy.multiply(SigmaPosterior, numpy.transpose(Array_Test[row,:]))
    SigmaPosterior = SigmaPosterior + inv(SigmaPrior_Local)
    SigmaPosterior = inv(SigmaPosterior)
    #print("SigmaPosterior")
    #print(SigmaPosterior)
    return SigmaPosterior

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def ReturnHighestValuesIndex(value, array):
    index=0
    while index<array.shape[0]:
        #if array[index] == value:
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

#print("")
#print("Part2")
#print("")

#print("Step 1 :")
SigmaPriorInverse = LambdaDotIdentityMatrix + numpy.multiply(1/Sigma2, XTransposeX)
SigmaPrior = inv(SigmaPriorInverse)
#print("SigmaPrior")
#print(SigmaPrior)

Test_Set = X_test
OutputArray = []


indexGlobal = 0
while indexGlobal < 10:
    # Part2
    #print("SigmaPrior")
    #print(SigmaPrior)
    Sigma0Array = SetSigmaOs(SigmaPrior, Test_Set);
    #print("Sigma0Array")
    #print(Sigma0Array)

    # Part3
    #print("Part 3 : ")
    Sigma0ArraySorted = numpy.sort(Sigma0Array,axis=0)
    HighestValueIndex = ReturnHighestValuesIndex(Sigma0ArraySorted[-1], Sigma0Array)
    SavedHighestValueIndex = HighestValueIndex
    print(HighestValueIndex)
    #HighestValueIndex = GetCorrectiveValue(HighestValueIndex)
    HighestValueIndex += indexGlobal
    ToBeSubstracted = 0
    for e in OutputArray:
        if e > HighestValueIndex:
            ToBeSubstracted += 1
    HighestValueIndex = HighestValueIndex - ToBeSubstracted
    OutputArray.append(HighestValueIndex)
    # Part4
    #print("Part 4 : ")
    #print("TestSet size before : "+str(Test_Set.shape))
    ##print(Test_Set)
    Test_Set = numpy.delete(Test_Set, (SavedHighestValueIndex), axis=0)
    #print("TestSet size after : "+str(Test_Set.shape))
    #print(Test_Set)

    # Part5
    SigmaPosterior = CalculPosterior(SigmaPrior,HighestValueIndex, X_train)
    SigmaPrior = SigmaPosterior
    # Loop
    indexGlobal += 1

#print("active :")
#print(OutputArray)

nameOfThePart2File = "active_"+str(int(Lambda))+"_"+str(int(Sigma2))+".csv"
print(nameOfThePart2File)

with open(nameOfThePart2File, 'wb') as csvfile2:
    spamwriter2 = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter2.writerow(OutputArray)


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
