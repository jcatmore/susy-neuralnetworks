import sys
import pickle

import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn import preprocessing

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from CommonTools import reconstructionError,reconstructionErrorByFeature,extractVariables,buildArrays,makeMetrics,areaUnderROC
from featuresLists import susyFeatures, susyWeights

###############################################
# MAIN PROGRAM

runTraining = True
nEvents = 20000

normalInputFile = open("SM_background.csv","r")
signalInputFile = open("SUSY_signal.csv","r")

# Get the data
normalData = extractVariables(normalInputFile,2*nEvents)
signalData = extractVariables(signalInputFile,2*nEvents)

# Assemble data arrays
cut = np.ones(2*nEvents)

X_train_bg = buildArrays(susyFeatures,cut,normalData,0,nEvents,"TRAINING SAMPLE (background)")
X_train_sig = buildArrays(susyFeatures,cut,signalData,0,nEvents,"TRAINING SAMPLE (signal)")
X_train = np.concatenate((X_train_bg,X_train_sig),0)

X_test_bg = buildArrays(susyFeatures,cut,normalData,nEvents,nEvents,"TESTING SAMPLE (background)")
X_test_sig = buildArrays(susyFeatures,cut,signalData,nEvents,nEvents,"TESTING SAMPLE (signal)")
X_test = np.concatenate((X_test_bg,X_test_sig),0)

Y_bg = np.zeros(nEvents).reshape(nEvents,1)
Y_sig = np.ones(nEvents).reshape(nEvents,1)
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
                    Layer("Rectifier", units=36),
                    Layer("Softmax")],
            learning_rate=0.01,
            batch_size = 100,
            learning_rule = "momentum",
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
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_train[nEvents:2*nEvents]==1.0)/nEvents
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_train[nEvents:2*nEvents]==0.0)/nEvents
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_train[0:nEvents]==1.0)/nEvents
print "  Background identified as background (%): ",100.0*np.sum(pred_train[0:nEvents]==0.0)/nEvents
print ""
print "Testing sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_test[nEvents:2*nEvents]==1.0)/nEvents
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_test[nEvents:2*nEvents]==0.0)/nEvents
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_test[0:nEvents]==1.0)/nEvents
print "  Background identified as background (%): ",100.0*np.sum(pred_test[0:nEvents]==0.0)/nEvents

# Plotting - probabilities
#print probabilities_train[(Y==0.0).reshape(2*nEvents,)]

figA, axsA = plt.subplots(2, 1)
ax1, ax2 = axsA.ravel()
for ax in ax1, ax2:
    ax.set_ylabel("Events")
    ax.set_xlabel("NN signal probability")
ax1.hist(probabilities_train[(Y==0.0).reshape(2*nEvents,)][:,1], 100, (-0.5,1.5), facecolor='blue', alpha=0.4, histtype='stepfilled')
ax1.hist(probabilities_train[(Y==1.0).reshape(2*nEvents,)][:,1], 100, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')
ax2.hist(probabilities_test[(Y==0.0).reshape(2*nEvents,)][:,1], 100, (-0.5,1.5), facecolor='green', alpha=0.4, histtype='stepfilled')
ax2.hist(probabilities_test[(Y==1.0).reshape(2*nEvents,)][:,1], 100, (-0.5,1.5), facecolor='red', alpha=0.4, histtype='stepfilled')


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











