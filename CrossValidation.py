import sys
import pickle

import numpy as np
from sknn.mlp import Regressor, Classifier, Layer
from sklearn import preprocessing

from ROOT import gROOT, TFile

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from CommonTools import buildArraysFromROOT
from featuresLists import susyFeaturesNtup, susyWeightsNtup

###############################################
# MAIN PROGRAM

nSignalEvents = 20000
nBackgroundEvents = 20000
manualSearch = False
nRandomPoints = 15

# Input from command line
runAE = bool(int(sys.argv[1]))
runCL = bool(int(sys.argv[2]))

# Hyper parameter ranges
#hiddenUnits_ae = [6,12,18,24,30,36,72,144,360]
hiddenUnits_ae = [2,4,6,8,10,12]
#hiddenUnits_cl = [36,72,144,360,720]
hiddenUnits_cl = [72,720,860,1000]
#learning_rate = [0.001,0.005,0.01,0.02,0.1]
learning_rate_ae = [0.01]
learning_rate_cl = [0.01,0.005,0.001]
#n_iter = [10,50,100,200,500]
n_iter_ae = [10,20,50]
n_iter_cl = [50,100,200]

hyperparameterSets = []
nValidationCycles = 0

if (manualSearch==False):
    # Generate n_validationCycles hyperparameter sets at random
    i_hiddenUnits_ae = np.random.randint(0,len(hiddenUnits_ae),nRandomPoints)
    i_hiddenUnits_cl = np.random.randint(0,len(hiddenUnits_cl),nRandomPoints)
    i_learningRate_ae = np.random.randint(0,len(learning_rate_ae),nRandomPoints)
    i_learningRate_cl = np.random.randint(0,len(learning_rate_cl),nRandomPoints)
    i_nIter_ae = np.random.randint(0,len(n_iter_ae),nRandomPoints)
    i_nIter_cl = np.random.randint(0,len(n_iter_cl),nRandomPoints)
    for iter in range(0,nRandomPoints):
        set = [hiddenUnits_ae[i_hiddenUnits_ae[iter]],
               hiddenUnits_cl[i_hiddenUnits_cl[iter]],
               learning_rate_ae[i_learningRate_ae[iter]],
               learning_rate_cl[i_learningRate_cl[iter]],
               n_iter_ae[i_nIter_ae[iter]],
               n_iter_cl[i_nIter_cl[iter]]]
        hyperparameterSets.append(set)

if (manualSearch==True):
    # List required combinations
    hyperparameterSets.append([10,0,0.01,0,5,0])
    hyperparameterSets.append([10,0,0.01,0,20,0])
    hyperparameterSets.append([10,0,0.01,0,100,0])
    hyperparameterSets.append([10,0,0.01,0,200,0])
    hyperparameterSets.append([10,0,0.01,0,500,0])

# Input file and TTree
inputFile = TFile("TMVA_tree.root","read")
tree = inputFile.Get("TMVA_tree")

# Selections
cutBackground = "isSignal==0"
cutSignal = "isSignal==1"

# Assemble data arrays
# Training
X_train_background = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,0,nBackgroundEvents,"TRAINING SAMPLE (background)")
X_train_signal = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,0,nSignalEvents,"TRAINING SAMPLE (signal)")
X_train = np.concatenate((X_train_background,X_train_signal),0)
# Testing
X_test_background = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,nBackgroundEvents,nBackgroundEvents+nSignalEvents,"TESTING SAMPLE (background)")
X_test_signal = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,nBackgroundEvents,nBackgroundEvents+nSignalEvents,"TESTING SAMPLE (signal)")
X_test = np.concatenate((X_test_background,X_test_signal),0)
# Target/ground truth
Y_background = np.zeros(nBackgroundEvents).reshape(nBackgroundEvents,1)
Y_signal = np.ones(nSignalEvents).reshape(nSignalEvents,1)
Y = np.concatenate((Y_background,Y_signal),0)

# Feature scaling - has to be done separately for the autoencoder and the classifier
# to avoid contaminating the autoencoder training sample with information about the signal
# Autoencoder...
scaler_autoencoder = preprocessing.MinMaxScaler()
X_train_background = scaler_autoencoder.fit_transform(X_train_background)
X_test_background = scaler_autoencoder.transform(X_test_background)
X_test_signal = scaler_autoencoder.transform(X_test_signal)
# Classifier
scaler_classifier = preprocessing.MinMaxScaler()
X_train = scaler_classifier.fit_transform(X_train)
X_test = scaler_classifier.transform(X_test)

# Set target equal to input for the autoencoder
Y_autoencoder = X_train_background

# Cross-validation loop
counter = 0
for set in hyperparameterSets:
    print "Cross validation step ",counter
    hiddenUnitsAE = set[0]
    hiddenUnitsCL = set[1]
    learningRateAE = set[2]
    learningRateCL = set[3]
    nCyclesAE = set[4]
    nCyclesCL = set[5]
    outputFileNameAE = 'trained_fine/ae_'+str(hiddenUnitsAE)+'_'+str(learningRateAE)+'_'+str(nCyclesAE)+'.pkl'
    outputFileNameCL = 'trained_fine/cl_'+str(hiddenUnitsCL)+'_'+str(learningRateCL)+'_'+str(nCyclesCL)+'.pkl'
    
    # AUTOENCODER
    if (runAE==True):
        nn = Regressor(
                       layers=[
                               Layer("Rectifier", units=hiddenUnitsAE),
                               Layer("Linear")],
                       learning_rate=learningRateAE,
                       n_iter=nCyclesAE)
        nn.fit(X_train_background,Y_autoencoder)
        pickle.dump(nn, open(outputFileNameAE, 'wb'))

    # CLASSIFIER
    if (runCL==True):
        nn = Classifier(
                        layers=[
                                Layer("Rectifier", units=hiddenUnitsCL),
                                Layer("Softmax")],
                        learning_rate=learningRateCL,
                        n_iter=nCyclesCL)
        nn.fit(X_train,Y)
        pickle.dump(nn, open(outputFileNameCL, 'wb'))

    counter = counter+1










