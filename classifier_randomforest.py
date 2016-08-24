import sys
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from ROOT import gROOT, TFile

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from CommonTools import reconstructionError,reconstructionErrorByFeature,buildArraysFromROOT,makeMetrics,areaUnderROC
from featuresLists import susyFeaturesNtup, susyWeightsNtup

###############################################
# MAIN PROGRAM

runTraining = True
nBackgroundEvents = 50000
nSignalEvents = 50000

# Selections
cutBackground = "isSignal==0"
cutSignal = "isSignal==1"

# Input file and TTree
inputFile = TFile("TMVA_tree.root","read")
tree = inputFile.Get("TMVA_tree")

# Build data arrays
X_train_bg = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,0,nBackgroundEvents,"TRAINING SAMPLE (background)")
W_bg = buildArraysFromROOT(tree,susyWeightsNtup,cutBackground,0,nBackgroundEvents,"EVENT WEIGHTS (background)").reshape(X_train_bg.shape[0])
X_train_sig = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,0,nSignalEvents,"TRAINING SAMPLE (signal)")
W_sig = buildArraysFromROOT(tree,susyWeightsNtup,cutSignal,0,nSignalEvents,"EVENT WEIGHTS (signal)").reshape(X_train_sig.shape[0])
X_train = np.concatenate((X_train_bg,X_train_sig),0)
W = np.concatenate((W_bg,W_sig),0)

X_test_bg = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,nBackgroundEvents,nBackgroundEvents,"TESTING SAMPLE (background)")
X_test_sig = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,nSignalEvents,nSignalEvents,"TESTING SAMPLE (signal)")
X_test = np.concatenate((X_test_bg,X_test_sig),0)

Y_bg = np.zeros(nBackgroundEvents).reshape(nBackgroundEvents,1)
Y_sig = np.ones(nSignalEvents).reshape(nSignalEvents,1)
Y = np.concatenate((Y_bg,Y_sig),0)

# Feature scaling
#min_max_scaler = preprocessing.MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.transform(X_test)

#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(X_train, Y.ravel())

clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, Y.ravel())

# Testing
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
probabilities_train = clf.predict_proba(X_train)
probabilities_test = clf.predict_proba(X_test)

print "Training sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_train[nBackgroundEvents:nBackgroundEvents+nSignalEvents]==1.0)/nSignalEvents
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_train[nBackgroundEvents:nBackgroundEvents+nSignalEvents]==0.0)/nSignalEvents
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_train[0:nBackgroundEvents]==1.0)/nBackgroundEvents
print "  Background identified as background (%): ",100.0*np.sum(pred_train[0:nBackgroundEvents]==0.0)/nBackgroundEvents
print ""
print "Testing sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_test[nBackgroundEvents:nBackgroundEvents+nSignalEvents]==1.0)/nSignalEvents
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_test[nBackgroundEvents:nBackgroundEvents+nSignalEvents]==0.0)/nSignalEvents
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_test[0:nBackgroundEvents]==1.0)/nBackgroundEvents
print "  Background identified as background (%): ",100.0*np.sum(pred_test[0:nBackgroundEvents]==0.0)/nBackgroundEvents

# Plotting - probabilities
#print probabilities_train[(Y==0.0).reshape(2*nEvents,)]

figA, axsA = plt.subplots(2, 1)
ax1, ax2 = axsA.ravel()
for ax in ax1, ax2:
    ax.set_ylabel("Events")
    ax.set_xlabel("NN signal probability")
ax1.hist(probabilities_train[(Y==0.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='blue', alpha=0.4, histtype='stepfilled')
ax1.hist(probabilities_train[(Y==1.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')
ax2.hist(probabilities_test[(Y==0.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='green', alpha=0.4, histtype='stepfilled')
ax2.hist(probabilities_test[(Y==1.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')


# Plotting - performance curves
# ROC
fpr, tpr, thresholds = roc_curve(Y, probabilities_test[:,1], pos_label=1)
print "Area under ROC = ",roc_auc_score(Y, probabilities_test[:,1])
figB, axsB = plt.subplots(1,2)
axB1,axB2 = axsB.ravel()
axB1.plot(fpr, tpr, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Anomaly Rate')
axB1.set_ylabel('True Anomaly Rate')
# Precision/recall
precision, recall, threshold = precision_recall_curve(Y, probabilities_test[:,1])
axB2.plot(recall, precision, label='Recall vs precision')
axB1.plot([0, 1], [0, 1], 'k--')
axB2.set_xlim([0.0, 1.0])
axB2.set_ylim([0.0, 1.05])
axB2.set_xlabel('Precision')
axB2.set_ylabel('Recall')

plt.show()


## Plotting - raw variables
#figA, axsA = plt.subplots(4, 4)
#nColumn = 0
#for axA in axsA.ravel():
#    axA.hist(X_train[:,nColumn], 250, facecolor='blue', alpha=0.4, histtype='stepfilled', normed=True)
#    axA.hist(X_anomaly[:,nColumn][rec_errors_anomaly > -3.0], 250, facecolor='red', alpha=0.4, histtype='stepfilled', normed=True)
#    #axA.hist(X_anomaly[:,nColumn], 250, facecolor='red', alpha=0.4, histtype='stepfilled', normed=True)
#    nColumn = nColumn+1











