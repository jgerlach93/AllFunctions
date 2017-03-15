from datetime import datetime
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy import *
from scipy import *
import pandas as pd
import portfolio_functions as pf
import sklearn as sk
from sklearn.svm import SVC
import dropbox

from sklearn.utils import column_or_1d
import portfolio_functions as pf
from pydatastream import Datastream

from keras.layers import Activation, Dense, LSTM
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
import graphlab as gl
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import rbm, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import gaussian_process
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV


##################################################
"""
Machine-Learning Algorithms Library
gdemos@ethz.ch // Mar.2017 // Zürich Switzerland.
"""
##################################################

###############################################################
#### DATA FUNCTIONS ####
###############################################################

def getTrainingSampleAndTestSample(data, labels):
    """
    Import as numpy arrays
    """

    # Munge data
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data,
                                                                         labels,
                                                                         test_size=0.1)
    print('Done')
    return X_train, X_test, y_train, y_test


###############################################################
def loadDataNumerai(v1=True):
    if v1 == True:
        path = '/Users/demos/Desktop/Kaggle/numerai_datasets/'
    else:
        path = '/Users/demos/Desktop/Kaggle/numerai_datasets2/'

    training_data = pd.read_csv(path + 'numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv(path + 'numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target']
    X = training_data.drop('target', axis=1)
    t_id = prediction_data['t_id']
    x_prediction = prediction_data.drop('t_id', axis=1)

    return training_data, prediction_data, X, Y, x_prediction


###############################################################
#### CLASSIFIERS ####
###############################################################

def estimateSGDAndPredict(X_test, X_train, y_test, y_train):
    """ coin flipping ==> better results """

    # CREATE, TRAIN AND PREDICT
    clf = linear_model.SGDClassifier(alpha=0.001, n_iter=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    resuSGD = clf.predict(X_test)

    DF = pd.DataFrame(resuSGD, columns=['SGD'])
    DF['true'] = y_test

    return DF


###############################################################
def kMeanClassifier(X_test, X_train, y_test, y_train):
    """Top one so far yielding acc=0.70"""

    # CREATE, TRAIN AND PREDICT
    clf = KNeighborsClassifier(3, p=2, n_jobs=-1)
    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['Kmean'])
    DF['true'] = y_test

    return DF


###############################################################
def kMeanClassifier2(X_test, X_train, y_test, y_train):
    """Top one so far yielding acc=0.70"""

    # CREATE, TRAIN AND PREDICT
    clf = KNeighborsClassifier(3, p=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['Kmean'])
    DF['true'] = y_test

    return DF


###############################################################
def kMeanClassifier3(X_test, X_train, y_test, y_train):
    """Top one so far yielding acc=0.70"""

    # CREATE, TRAIN AND PREDICT
    clf = KNeighborsClassifier(3, p=0.5, n_jobs=-1, leaf_size=90,
                               weights='distance',
                               metric='mahalanobis', V=0.75)
    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['Kmean'])
    DF['true'] = y_test

    return DF


###############################################################
def nearestNeighbourClassifier(X_test, X_train, y_test, y_train):
    """Never Tested-it"""

    # CREATE, TRAIN AND PREDICT
    clf = RadiusNeighborsClassifier(3, p=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['NNb'])
    DF['true'] = y_test

    return DF


###############################################################
def GaussianProccesClassifier(X_test, X_train, y_test, y_train):
    clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['GP'])
    DF['true'] = y_test

    return DF


###############################################################
def neuralNetsClassifier(X_test, X_train, y_test, y_train):
    """Learn How To Tune"""

    # CREATE, TRAIN AND PREDICT
    clf = MLPClassifier(solver='adam', alpha=0.001,
                        activation='tanh', batch_size=1,
                        hidden_layer_sizes=(100, 15), random_state=10,
                        beta_1=0.009, beta_2=0.01, max_iter=2000)

    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['NN'])
    DF['true'] = y_test

    return DF


###############################################################
def RandomTreeClass(X_test, X_train, y_test, y_train):
    """TAKES A LONG TIME"""

    # CREATE, TRAIN AND PREDICT
    clf = RandomForestClassifier(max_depth=500,
                                 n_estimators=200,
                                 max_features=50)
    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['Rtree'])
    DF['true'] = y_test

    return DF


###############################################################
def SVCmitRBF(X_test, X_train, y_test, y_train):
    """TAKES A LONG TIME"""

    # CREATE, TRAIN AND PREDICT
    clf = SVC(gamma=3.5, C=1)

    clf.fit(X_train, y_train)
    resu = clf.predict(X_test)

    DF = pd.DataFrame(resu, columns=['SVC'])
    DF['true'] = y_test

    return DF


###############################################################
#### POOLING OF CLASSIFIERS ####
###############################################################

def Estimate_Several_Classificators(X_test, X_train, y_test, y_train):
    names = ["QDA", "Nearest Neighbors", "RBF2 SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "AdaBoost",
             "Naive Bayes"]

    classifiers = [
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(4),
        SVC(gamma=1, C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB()]

    DF = pd.DataFrame()

    for classifieri in range(len(classifiers)):
        print('classifying %s' % classifieri)
        clf = classifiers[classifieri]
        clf.fit(X_train, y_train)
        resu = clf.predict(X_test)

        D = pd.DataFrame(resu, columns=[str(classifieri)])
        DF = pd.concat([DF, D], axis=1)

    DF['true'] = y_test

    return DF


###############################################################
def send2Numerai(prediction_data, clf):
    """ Predict data and save on CSV format """

    trueDataToForecast = prediction_data[prediction_data.columns[1:]].values
    print(shape(trueDataToForecast))

    resuProbKmeanTrue = clf.predict_proba(trueDataToForecast)

    results = resuProbKmeanTrue[:, 1].copy()
    results_df = pd.DataFrame(data={'probability': results})
    joined = pd.DataFrame(prediction_data['t_id']).join(results_df)

    joined.to_csv("GDpredictionsData_nn.csv", index=False)
