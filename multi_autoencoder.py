import sys
import pickle

from ROOT import gROOT, gDirectory, TFile, TEventList, TCut

import numpy as np
from sknn.mlp import Regressor, Layer
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn import metrics

from CommonTools import reconstructionError,reconstructionErrorByFeature,buildArraysFromROOT,makeMetrics,areaUnderROC
from featuresLists import susyFeaturesNtup, susyWeightsNtup

###############################################
# MAIN PROGRAM

runTraining = False

# Input file and TTree
inputFile = TFile("TMVA_tree.root","read")
tree = inputFile.Get("TMVA_tree")

# Full background samples
nAllBackgroundEvents = 100000
cutBackground = "isSignal==0"
X_train_bg = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,0,nAllBackgroundEvents,"TRAINING SAMPLE (background)")
X_test_bg_all = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,nAllBackgroundEvents,nAllBackgroundEvents,"TESTING SAMPLE (background)")

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train_bg = min_max_scaler.fit_transform(X_train_bg)
X_test_bg_all = min_max_scaler.transform(X_test_bg_all)

# Set target equal to input - auto-encoder
Y_train = X_train_bg
    
# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network
if runTraining:
    print "Starting neural network training"
    nn = Regressor(
                    layers=[
                            Layer("Rectifier", units=30),
                            Layer("Linear")],
                    learning_rate=0.01,
                    batch_size = 100,
                    #learning_rule = "momentum",
                    n_iter=500)
    # Training
    nn.fit(X_train_bg,Y_train)
    pickle.dump(nn, open('autoencoder.pkl', 'wb'))
if not runTraining:
    nn = pickle.load(open('autoencoder.pkl', 'rb'))

cutList = ["isSignal==1 && massSplit<199",
           "isSignal==1 && massSplit>199 && massSplit<299",
           "isSignal==1 && massSplit>299 && massSplit<399",
           "isSignal==1 && massSplit>399 && massSplit<499",
           "isSignal==1 && massSplit>499 && massSplit<599",
           "isSignal==1 && massSplit>599 && massSplit<699"]

# Set up the axes for plotting
figROC, axsROC = plt.subplots(2,3)
axesROC = axsROC.ravel()
figRecErr, axsRecErr = plt.subplots(2,3)
axesRecErr = axsRecErr.ravel()

# MAIN LOOP OVER SIGNAL SAMPLES
axesCounter = 0
for cut in cutList:
    
    # Selections and numbers of events
    cutSignal = cut
    tcut = TCut(cutSignal)
    tree.Draw(">>eventList",tcut)
    eventList = TEventList()
    eventList = gDirectory.Get("eventList")
    nSignalEvents = eventList.GetN()/2
    nBackgroundEvents = nSignalEvents
    
    print "====================================="
    print "Selection: ",cutSignal, "with ",nSignalEvents," signal events and ",nBackgroundEvents," background events"
    
    # Build data arrays
    X_test_bg = X_test_bg_all[0:nBackgroundEvents,:]
    X_test_sig = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,nSignalEvents,nSignalEvents,"TESTING SAMPLE (signal)")
    X_test_sig = min_max_scaler.transform(X_test_sig)

    # Testing
    predicted_diff = nn.predict(X_test_bg)
    predicted_signal = nn.predict(X_test_sig)

    # Reconstruction error
    rec_errors_diff = reconstructionError(X_test_bg,predicted_diff)
    rec_errors_sig = reconstructionError(X_test_sig,predicted_signal)

    # Reconstruction errors by variable
    rec_errors_varwise_diff = reconstructionErrorByFeature(X_test_bg,predicted_diff)
    rec_errors_varwise_sig = reconstructionErrorByFeature(X_test_sig,predicted_signal)

    ## Plotting - performance curves
    ## ROC
    true_positive,false_positive,precisions,recalls,f1s = makeMetrics(200,rec_errors_sig,rec_errors_diff)
    print "Area under ROC = ",areaUnderROC(true_positive,false_positive)
    print ""
    print ""
    print ""
    axesROC[axesCounter].plot(false_positive, true_positive, label='ROC curve')
    axesROC[axesCounter].plot([0, 1], [0, 1], 'k--')
    axesROC[axesCounter].set_xlim([0.0, 1.0])
    axesROC[axesCounter].set_ylim([0.0, 1.05])
    axesROC[axesCounter].set_xlabel('False Anomaly Rate')
    axesROC[axesCounter].set_ylabel('True Anomaly Rate')

    # NN probabilities
    axesRecErr[axesCounter].set_ylabel("Events")
    axesRecErr[axesCounter].set_xlabel("log10(Reconstruction error)")
    #axesRecErr[axesCounter].hist(probabilities_train[(Y==0.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='blue', alpha=0.4, histtype='stepfilled')
    #axesRecErr[axesCounter].hist(probabilities_train[(Y==1.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')
    axesRecErr[axesCounter].hist(rec_errors_diff, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
    axesRecErr[axesCounter].hist(rec_errors_sig, 250, facecolor='red', alpha=0.4, histtype='stepfilled')
    
    axesCounter = axesCounter+1

plt.show()


## Plotting - reconstruction error per variable
#figD, axsD = plt.subplots(6, 6)
#nColumn = 0
#for axD in axsD.ravel():
#    axD.hist(rec_errors_varwise_diff[:,nColumn], 250, facecolor='green', alpha=0.4, histtype='stepfilled', normed=True)
#    axD.hist(rec_errors_varwise_sig[:,nColumn], 250, facecolor='red', alpha=0.4, histtype='stepfilled', normed=True)
#    nColumn = nColumn+1
#
## Plotting - raw variables
#figA, axsA = plt.subplots(6, 6)
#nColumn = 0
#for axA in axsA.ravel():
#    axA.hist(X_train[:,nColumn], 250, facecolor='blue', alpha=0.4, histtype='stepfilled')
#    #axA.hist(X_signal[:,nColumn][rec_errors_sig > -1.75], 250, facecolor='red', alpha=0.4, histtype='stepfilled', normed=True)
#    axA.hist(X_signal[:,nColumn], 250, facecolor='red', alpha=0.4, histtype='stepfilled')
#    nColumn = nColumn+1
#
## Plotting - labels
#labelsList = [val.GetName() for val in tree.GetListOfBranches() if val.GetName() in susyFeaturesNtup]
#figC, axsC = plt.subplots(6, 6)
#for axC in axsC.ravel():
#    if len(labelsList)==0:
#        break
#    label = labelsList[0]
#    axC.text(0.1, 0.5, label, fontsize=10)
#    axC.set_xticklabels([])
#    axC.set_yticklabels([])
#    axC.set_xticks([])
#    axC.set_yticks([])
#    labelsList.pop(0)









