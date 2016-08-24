import sys
import pickle

from ROOT import gROOT, TFile

import numpy as np
from sklearn import svm
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.externals import joblib

from CommonTools import reconstructionError,reconstructionErrorByFeature,buildArraysFromROOT,makeMetrics,areaUnderROC
from featuresLists import susyFeaturesNtup, susyWeightsNtup

###############################################
# MAIN PROGRAM

runTraining = False
nBackgroundEvents = 50000
nSignalEvents = 50000

# Selections
cutBackground = "isSignal==0"
cutSignal = "isSignal==1"

# Input file and TTree
inputFile = TFile("TMVA_tree.root","read")
tree = inputFile.Get("TMVA_tree")

# Assemble data arrays
X_train = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,0,nBackgroundEvents,"TRAINING SAMPLE (background only)")
W_train = buildArraysFromROOT(tree,susyWeightsNtup,cutBackground,0,nBackgroundEvents,"TRAINING SAMPLE WEIGHTS").reshape(X_train.shape[0])
X_test = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,nBackgroundEvents,nBackgroundEvents,"TESTING SAMPLE - background")
X_signal = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,0,nSignalEvents,"TESTING SAMPLE - signal")

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
X_signal = min_max_scaler.transform(X_signal)

# Results dictionary
results = {}

# One-class fitting with the SVM
nus = [nu / 20.0 for nu in range(1, 20)]
nus.append(0.99)
nus.insert(0,0.01)
#nus.insert(0,0.001)
for nu in nus:
    sys.stdout.write("Processing nu = %s        \r" % nu )
    sys.stdout.flush()
    outputName = 'susy_svm_'+str(nu)+'.pkl'
    if runTraining==True:
        clf = svm.OneClassSVM(nu=nu, kernel='rbf')
        clf.fit(X_train)
        joblib.dump(clf, outputName)
    if runTraining==False:
        clf = joblib.load(outputName)

    # Testing
    predicted_same = clf.predict(X_train)
    predicted_diff = clf.predict(X_test)
    predicted_signal = clf.predict(X_signal)

    distances_train = clf.decision_function(X_train)
    distances_test = clf.decision_function(X_test)
    distances_signal = clf.decision_function(X_signal)

    # Results filling
    trainingAnomalies = float(predicted_same[predicted_same==-1].size)/float(predicted_same.size)
    testingAnomalies = float(predicted_diff[predicted_diff==-1].size)/float(predicted_diff.size)
    signalAnomalies = float(predicted_signal[predicted_signal==-1].size)/float(predicted_signal.size)
    thisResult = [trainingAnomalies,testingAnomalies,signalAnomalies]
    results[nu] = thisResult

    # Plotting
    #fig, axs = plt.subplots(3, 1)
    #ax1, ax2, ax3 = axs.ravel()
    #for ax in ax1, ax2, ax3:
    #    ax.set_ylabel("Events")
    #    ax.set_xlabel("Distance from decision surface")
    #ax1.hist(distances_train, 250, facecolor='blue', alpha=0.4, histtype='stepfilled')
    #ax2.hist(distances_test, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
    #ax3.hist(distances_test, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
    #ax3.hist(distances_signal, 250, facecolor='red', alpha=0.4, histtype='stepfilled')
sys.stdout.write("\n")

# Results output
true_positive = []
false_positive = []
print "========================================"
print "nu\tTraining anomalies\tTesting anomalies\tSignal anomalies"
for nu in nus:
    print nu,"\t",results[nu][0],"\t\t\t",results[nu][1],"\t\t\t",results[nu][2]
    true_positive.append(results[nu][2])
    false_positive.append(results[nu][1])
print "========================================"

plt.plot(false_positive, true_positive, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Anomaly Rate')
plt.ylabel('True Anomaly Rate')
plt.show()