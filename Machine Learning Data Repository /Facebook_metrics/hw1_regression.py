import csv
import sys
import numpy
from numpy import genfromtxt
from numpy.linalg import inv

numpy.set_printoptions(threshold=numpy.nan)

Lambda = float(sys.argv[1])
Sigma2 = float(sys.argv[2])

print("Lambda = ")
print(Lambda)
print("Sigma2 = ")
print(Sigma2)

X_train = genfromtxt('X_train.csv', delimiter=',')
print("X_train = ")
print(X_train)

y_train = genfromtxt('y_train.csv', delimiter=',')
print("y_train = ")
print(y_train)

X_test = genfromtxt('X_test.csv', delimiter=',')
print("X_test = ")
print(X_test)

# Get the number of columns -> size of identity_matrix
# Shape returns a tuple (rows, columns)
columns = X_train.shape[1]

identity_matrix = numpy.identity(columns)

LambdaDotIdentityMatrix = numpy.multiply(identity_matrix,Lambda)
print("LambdaDotIdentityMatrix = ")
print(LambdaDotIdentityMatrix)

XTransposeX = numpy.transpose(X_train).dot(X_train)
print("XTransposeX")
print(XTransposeX)

Inverse = inv(LambdaDotIdentityMatrix+XTransposeX)
print("Inverse")
print(Inverse)

XtransposeY = numpy.transpose(X_train).dot(y_train)
print("XtransposeY")
print(XtransposeY)

wRR = Inverse.dot(XtransposeY)
print("wRR")
print(wRR)

nameOfThePart1File = "wRR_"+str(Lambda)+".csv"
print(nameOfThePart1File)

with open(nameOfThePart1File, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(numpy.transpose(wRR))

nameOfThePart2File = "active_"+str(Lambda)+"_"+str(Sigma2)+".csv"
print(nameOfThePart2File)

with open(nameOfThePart2File, 'wb') as csvfile2:
    spamwriter2 = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter2.writerow(wRR)



# print(numpy.transpose(X_train))

# f = open ( 'input.txt' , 'r')
# l = [ map(int,line.split(',')) for line in f ]
# print l

# with fileinput.input(files=('X_train.csv')) as f:
#     for line in f:
#         process(line)


#python hw1_regression.py lambda sigma2  y_train.csv X_test.csv
