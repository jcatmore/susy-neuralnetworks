import sys
import pickle

from ROOT import gROOT, TFile

import numpy as np
from sklearn.neighbors.kde import KernelDensity
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

# Kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(X_train)
densities_train = kde.score_samples(X_train)
densities_test = kde.score_samples(X_test)
densities_signal = kde.score_samples(X_signal)


# Density plotting
fig, axs = plt.subplots(3, 1)
ax1, ax2, ax3 = axs.ravel()
for ax in ax1, ax2, ax3:
    ax.set_ylabel("Events")
    ax.set_xlabel("Density")
ax1.hist(densities_train, 1000,  facecolor='blue', alpha=0.4, histtype='stepfilled')
ax2.hist(densities_train, 1000,  facecolor='blue', alpha=0.4, histtype='stepfilled')
ax2.hist(densities_test, 1000,   facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(densities_test, 1000,   range=[-100.0,100.0], facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(densities_signal, 1000, range=[-100.0,100.0], facecolor='red', alpha=0.4, histtype='stepfilled')

# ROC curve
true_positive,false_positive,precisions,recalls,f1s = makeMetrics(5000,densities_signal,densities_test,reverse=True)
print "Area under ROC = ",areaUnderROC(true_positive,false_positive)
figB, axsB = plt.subplots(1,2)
axB1,axB2 = axsB.ravel()
# ROC
axB1.plot(false_positive, true_positive, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Anomaly Rate')
axB1.set_ylabel('True Anomaly Rate')
# Precision, recall
axB2.plot(recalls, precisions, label='Precision-recall curve')
axB2.plot([0, 1.0], [0.5, 0.5], 'k--')
axB2.set_xlim([0.0, 1.0])
axB2.set_ylim([0.0, 1.05])
axB2.set_xlabel('Recall')
axB2.set_ylabel('Precision')

plt.show()

