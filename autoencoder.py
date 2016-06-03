import sys
import pickle

from ROOT import gROOT, TFile

import numpy as np
from sknn.mlp import Regressor, Layer
from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

from CommonTools import reconstructionError,reconstructionErrorByFeature,buildArraysFromROOT,makeMetrics,areaUnderROC
from featuresLists import susyFeaturesNtup, susyWeightsNtup

###############################################
# MAIN PROGRAM

runTraining = True
nBackgroundEvents = 20000
nSignalEvents = 20000

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

# Set target equal to input - replicator NN
Y_train = X_train

# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network
if runTraining:
    print "Starting neural network training"
    nn = Regressor(
                   layers=[
                           Layer("Rectifier", units=10),
                           Layer("Linear")],
                   learning_rate=0.01,
                   #batch_size = 10,
                   #learning_rule = "momentum",
                   n_iter=100)
    # Training
    nn.fit(X_train,Y_train)
    pickle.dump(nn, open('autoencoder.pkl', 'wb'))
if not runTraining:
    nn = pickle.load(open('autoencoder.pkl', 'rb'))


# Testing
predicted_same = nn.predict(X_train)
predicted_diff = nn.predict(X_test)
predicted_signal = nn.predict(X_signal)

# Reconstruction error
rec_errors_same = reconstructionError(X_train,predicted_same)
rec_errors_diff = reconstructionError(X_test,predicted_diff)
rec_errors_sig = reconstructionError(X_signal,predicted_signal)

# Reconstruction errors by variable
rec_errors_varwise_same = reconstructionErrorByFeature(X_train,predicted_same)
rec_errors_varwise_diff = reconstructionErrorByFeature(X_test,predicted_diff)
rec_errors_varwise_sig = reconstructionErrorByFeature(X_signal,predicted_signal)

# Plotting - reconstruction errors
fig, axs = plt.subplots(3, 1)
ax1, ax2, ax3 = axs.ravel()
for ax in ax1, ax2, ax3:
    ax.set_ylabel("Events")
    ax.set_xlabel("log10(Reconstruction error)")
ax1.hist(rec_errors_same, 250, facecolor='blue', alpha=0.4, histtype='stepfilled')
ax2.hist(rec_errors_diff, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_diff, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_sig, 250, facecolor='red', alpha=0.4, histtype='stepfilled')

# Plotting - reconstruction error per variable
figD, axsD = plt.subplots(6, 6)
nColumn = 0
for axD in axsD.ravel():
    axD.hist(rec_errors_varwise_diff[:,nColumn], 250, facecolor='green', alpha=0.4, histtype='stepfilled', normed=True)
    axD.hist(rec_errors_varwise_sig[:,nColumn], 250, facecolor='red', alpha=0.4, histtype='stepfilled', normed=True)
    nColumn = nColumn+1

# Plotting - raw variables
figA, axsA = plt.subplots(6, 6)
nColumn = 0
for axA in axsA.ravel():
    axA.hist(X_train[:,nColumn], 250, facecolor='blue', alpha=0.4, histtype='stepfilled', normed=True)
    axA.hist(X_signal[:,nColumn][rec_errors_sig > -1.75], 250, facecolor='red', alpha=0.4, histtype='stepfilled', normed=True)
    #axA.hist(X_signal[:,nColumn], 250, facecolor='red', alpha=0.4, histtype='stepfilled')
    nColumn = nColumn+1

# Plotting - labels
labelsList = [val.GetName() for val in tree.GetListOfBranches() if val.GetName() in susyFeaturesNtup]
figC, axsC = plt.subplots(6, 6)
for axC in axsC.ravel():
    if len(labelsList)==0:
        break
    label = labelsList[0]
    axC.text(0.1, 0.5, label, fontsize=10)
    axC.set_xticklabels([])
    axC.set_yticklabels([])
    axC.set_xticks([])
    axC.set_yticks([])
    labelsList.pop(0)

# Plotting - performance curves
true_positive,false_positive,precisions,recalls,f1s = makeMetrics(200,rec_errors_sig,rec_errors_diff)
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









