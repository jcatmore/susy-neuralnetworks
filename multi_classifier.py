import sys
import pickle

import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn import preprocessing

from ROOT import gROOT, gDirectory, TFile, TEventList, TCut

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

# Input file and TTree
inputFile = TFile("TMVA_tree.root","read")
tree = inputFile.Get("TMVA_tree")

# Full background samples
nAllBackgroundEvents = 100000
cutBackground = "isSignal==0"
X_train_bg_all = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,0,nAllBackgroundEvents,"TRAINING SAMPLE (background)")
#W_bg_all = buildArraysFromROOT(tree,susyWeightsNtup,cutBackground,0,nAllBackgroundEvents,"EVENT WEIGHTS (background)").reshape(X_train_bg_all.shape[0])
X_test_bg_all = buildArraysFromROOT(tree,susyFeaturesNtup,cutBackground,nAllBackgroundEvents,nAllBackgroundEvents,"TESTING SAMPLE (background)")

cutList = ["isSignal==1 && massSplit<199",
           "isSignal==1 && massSplit>199 && massSplit<299",
           "isSignal==1 && massSplit>299 && massSplit<399",
           "isSignal==1 && massSplit>399 && massSplit<499",
           "isSignal==1 && massSplit>499 && massSplit<599",
           "isSignal==1 && massSplit>599 && massSplit<699"]

#cutSpcl = "isSignal==1 && massSplit<199"

# Set up the axes for plotting
figROC, axsROC = plt.subplots(2,3)
axesROC = axsROC.ravel()
figProb, axsProb = plt.subplots(2,3)
axesProb = axsProb.ravel()

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
    X_train_sig = buildArraysFromROOT(tree,susyFeaturesNtup,cutSignal,0,nSignalEvents,"TRAINING SAMPLE (signal)")
    X_train = np.concatenate((X_train_bg,X_train_sig),0)
    #W_sig = buildArraysFromROOT(tree,susyWeightsNtup,cutSignal,0,nSignalEvents,"EVENT WEIGHTS (signal)").reshape(X_train_sig.shape[0])
    #W = np.concatenate((W_bg,W_sig),0)

    X_test_bg = X_test_bg_all[0:nBackgroundEvents,:]
    X_test_sig = buildArraysFromROOT(tree,susyFeaturesNtup,cut,nSignalEvents,nSignalEvents,"TESTING SAMPLE (signal)")
    X_test = np.concatenate((X_test_bg,X_test_sig),0)

    Y_bg = np.zeros(nBackgroundEvents).reshape(nBackgroundEvents,1)
    Y_sig = np.ones(nSignalEvents).reshape(nSignalEvents,1)
    Y_sig = np.ones(nSignalEvents).reshape(nSignalEvents,1)
    Y = np.concatenate((Y_bg,Y_sig),0)
    Y = np.concatenate((Y_bg,Y_sig),0)

    # Feature scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # NEURAL NETWORK TRAINING AND TESTING
    # Set up neural network
    if runTraining:
        print "Starting neural network training"
        nn = Classifier(
                        layers=[
                                Layer("Rectifier", units=49),
                                Layer("Softmax")],
                        learning_rate=0.01,
                        batch_size = 100,
                        n_iter=100)
    # Training
    nn.fit(X_train,Y)
    pickle.dump(nn, open('nn_susy_classification.pkl', 'wb'))
    if not runTraining:
        nn = pickle.load(open('nn_susy_classification.pkl', 'rb'))

    # Testing
    pred_train = nn.predict(X_train)
    pred_test = nn.predict(X_test)
    probabilities_train = nn.predict_proba(X_train)
    probabilities_test = nn.predict_proba(X_test)

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

    ## Plotting - performance curves
    ## ROC
    fpr, tpr, thresholds = roc_curve(Y, probabilities_test[:,1], pos_label=1)
    auc = roc_auc_score(Y, probabilities_test[:,1])
    print "Area under ROC = ",auc
    print ""
    print ""
    print ""
    axesROC[axesCounter].plot(fpr, tpr, label='ROC curve')
    axesROC[axesCounter].plot([0, 1], [0, 1], 'k--')
    axesROC[axesCounter].set_xlim([0.0, 1.0])
    axesROC[axesCounter].set_ylim([0.0, 1.05])
    axesROC[axesCounter].set_xlabel('False Signal Rate')
    axesROC[axesCounter].set_ylabel('True Signal Rate')
    axesROC[axesCounter].text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)
    # Precision/recall
    #precision, recall, threshold = precision_recall_curve(Y, probabilities_test[:,1])
    #axB2.plot(recall, precision, label='Recall vs precision')
    #axB1.plot([0, 1], [0, 1], 'k--')
    #axB2.set_xlim([0.0, 1.0])
    #axB2.set_ylim([0.0, 1.05])
    #axB2.set_xlabel('Precision')
    #axB2.set_ylabel('Recall')
    
    # NN probabilities
    bins = np.linspace(-0.1, 1.1, 250)
    axesProb[axesCounter].set_ylabel("Events")
    axesProb[axesCounter].set_xlabel("NN signal probability")
    #axesProb[axesCounter].hist(probabilities_train[(Y==0.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='blue', alpha=0.4, histtype='stepfilled')
    #axesProb[axesCounter].hist(probabilities_train[(Y==1.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], 250, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')
    axesProb[axesCounter].hist(probabilities_test[(Y==0.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], bins, (-0.5,1.5), facecolor='green', alpha=0.4, histtype='stepfilled')
    axesProb[axesCounter].hist(probabilities_test[(Y==1.0).reshape(nBackgroundEvents+nSignalEvents,)][:,1], bins, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')

    axesCounter = axesCounter+1

plt.show()

