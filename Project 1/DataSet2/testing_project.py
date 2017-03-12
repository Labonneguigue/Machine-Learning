# This assumes you have a method called RidgeRegression and a method called ActiveLearning
from hw1_regression import *

# Used for estimating Mean Squared Error
from sklearn.model_selection import KFold

if __name__ == '__main__':

    lam = 1
    sigma2 = 2
    X_train = np.genfromtxt('X_train.csv', delimiter=',', skip_header=0, names=None)
    y_train = np.genfromtxt('y_train.csv', delimiter=',', skip_header=0, names=None)
    X_test = np.genfromtxt('X_test.csv', delimiter=',', skip_header=0, names=None)
    wRR = RidgeRegression(X_train, y_train, lam, centering=False, standarize=False)
    active =  ActiveLearning(X_train, X_test, lam, sigma2)

    # Estimating Mean Squared Error from training set, using 10-fold cv
    mean_squared_error = 0.0
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X_train):
        X_traincv, X_testcv = X_train[train_index], X_train[test_index]
        y_traincv, y_testcv = y_train[train_index], y_train[test_index]
        w = RidgeRegression(X_traincv, y_traincv, lam, centering=False, standarize=False)

        mean_squared_error += sum((y_testcv - np.dot(X_testcv, w))**2) / len(y_testcv)
    mean_squared_error = mean_squared_error / 10
        

    print "-------------------------"
    print "Dataset 1"
    print "-------------------------"
    print "Mean squared error: ", 
    print mean_squared_error
    print "wRR: ",
    print wRR
    print "Active Learning Indexes: ",
    print active

    X_train = np.genfromtxt('X_train1.csv', delimiter=',', skip_header=0, names=None)
    y_train = np.genfromtxt('y_train1.csv', delimiter=',', skip_header=0, names=None)
    X_test = np.genfromtxt('X_test1.csv', delimiter=',', skip_header=0, names=None)
    wRR = RidgeRegression(X_train, y_train, lam, centering=False, standarize=False)
    active =  ActiveLearning(X_train, X_test, lam, sigma2)

    # Estimating Mean Squared Error from training set, using 10-fold cv
    mean_squared_error = 0.0
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X_train):
        X_traincv, X_testcv = X_train[train_index], X_train[test_index]
        y_traincv, y_testcv = y_train[train_index], y_train[test_index]
        w = RidgeRegression(X_traincv, y_traincv, lam, centering=False, standarize=False)

        mean_squared_error += sum((y_testcv - np.dot(X_testcv, w))**2) / len(y_testcv)
    mean_squared_error = mean_squared_error / 10
        
    print "-------------------------"
    print "Dataset 2"
    print "-------------------------"
    print "Mean squared error: ", 
    print mean_squared_error
    print "wRR: ",
    print wRR
    print "Active Learning Indexes: ",
    print active