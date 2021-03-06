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
X_train_bg_all = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,0,nAllBackgroundEvents,"TRAINING SAMPLE (background)")
X_test_bg_all = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,nAllBackgroundEvents,nAllBackgroundEvents,"TESTING SAMPLE (background)")

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train_bg_all = min_max_scaler.fit_transform(X_train_bg_all)
X_test_bg_all = min_max_scaler.transform(X_test_bg_all)

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
    X_train_bg = X_train_bg_all[0:nBackgroundEvents,:]
    X_test_bg = X_test_bg_all[0:nBackgroundEvents,:]
    X_test_sig = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,nSignalEvents,nSignalEvents,"TESTING SAMPLE (signal)")
    X_test_sig = min_max_scaler.transform(X_test_sig)

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
                       n_iter=100)
                       #valid_size=0.25)
        # Training
        nn.fit(X_train_bg,Y_train)
        pickle.dump(nn, open('autoencoder.pkl', 'wb'))
    if not runTraining:
        nn = pickle.load(open('autoencoder.pkl', 'rb'))

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
    true_positive,false_positive,precisions,recalls,f1s = makeMetrics(2000,rec_errors_sig,rec_errors_diff)
    auc = areaUnderROC(true_positive,false_positive)
    print "Area under ROC = ",auc
    print ""
    print ""
    print ""
    axesROC[axesCounter].plot(false_positive, true_positive, label='ROC curve')
    axesROC[axesCounter].plot([0, 1], [0, 1], 'k--')
    axesROC[axesCounter].set_xlim([0.0, 1.0])
    axesROC[axesCounter].set_ylim([0.0, 1.05])
    axesROC[axesCounter].set_xlabel('False Anomaly Rate')
    axesROC[axesCounter].set_ylabel('True Anomaly Rate')
    axesROC[axesCounter].text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)


    # NN probabilities
    bins = np.linspace(-3.0, 3.0, 250)
    axesRecErr[axesCounter].set_ylabel("Events")
    axesRecErr[axesCounter].set_xlabel("log10(Reconstruction error)")
    #axesRecErr[axesCounter].hist(probabilities_train[(Y==0.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='blue', alpha=0.4, histtype='stepfilled')
    #axesRecErr[axesCounter].hist(probabilities_train[(Y==1.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')
    axesRecErr[axesCounter].hist(rec_errors_diff, bins=bins, facecolor='green', alpha=0.4, histtype='stepfilled')
    axesRecErr[axesCounter].hist(rec_errors_sig, bins=bins, facecolor='red', alpha=0.4, histtype='stepfilled')
    
    axesCounter = axesCounter+1

plt.show()










