import sys
import os
import pickle

import numpy as np
from sknn.mlp import Regressor, Classifier, Layer
from sklearn import preprocessing

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from CommonTools import reconstructionError,extractVariables,buildArrays,makeMetrics,areaUnderROC
from featuresLists import susyFeatures, susyWeights

###############################################
# MAIN PROGRAM

nSignalEvents = 20000
nBackgroundEvents = 20000

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

# Cross-validation loop
bestAE = []
bestCL = []
bestAEScore = -1.0
bestCLScore = -1.0

for fn in os.listdir('trained/'):
    print "Processing ",fn
    splitname = fn.split("_")
    type = splitname[0]
    nUnits = splitname[1]
    learningRate = splitname[2]
    nCycles = (splitname[3]).split('.')[0]
    inputFile = 'trained/'+fn
    
    if (type=='ae'):
        nn = pickle.load(open(inputFile, 'rb'))
        reconstucted_background = nn.predict(X_test_background)
        reconstucted_signal = nn.predict(X_test_signal)
        errors_background = reconstructionError(X_test_background,reconstucted_background)
        errors_signal = reconstructionError(X_test_signal,reconstucted_signal)
        true_positive,false_positive,precisions,recalls,f1s = makeMetrics(500,errors_signal,errors_background)
        tmpAUC = areaUnderROC(true_positive,false_positive)
        outputTextReport.write(type+' '+nUnits+' '+learningRate+' '+nCycles+' '+str(tmpAUC)+'\n')
        if tmpAUC > bestAEScore:
            bestAEScore = tmpAUC
            bestAE = [tmpAUC,type,nUnits,learningRate,nCycles]

    if (type=='cl'):
        nn = pickle.load(open(inputFile, 'rb'))
        predicted = nn.predict(X_test)
        probabilities = nn.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(Y, probabilities[:,1], pos_label=1)
        tmpAUC = roc_auc_score(Y, probabilities[:,1])
        outputTextReport.write(type+' '+nUnits+' '+learningRate+' '+nCycles+' '+str(tmpAUC)+'\n')
        if tmpAUC > bestCLScore:
            bestCLScore = tmpAUC
            bestCL = [tmpAUC,type,nUnits,learningRate,nCycles]

print "Best parameters for auto-encoder = ",bestAE
print "Best parameters for classifier = ",bestCL






