import numpy as np
import sys
from sklearn.preprocessing import Imputer

###############################################
# Reconstruction errors
def reconstructionError(test,predicted):
    deltas = test-predicted
    deltas_sq = np.power(deltas,2)
    deltas_sum = np.sum(deltas_sq,1)
    #rec_errors = deltas_sum/test.shape[0]
    rec_errors = np.log10(deltas_sum)
    return rec_errors
###############################################

###############################################
# Reconstruction errors by feature
def reconstructionErrorByFeature(test,predicted):
    deltas = test-predicted
    deltas_sq = np.power(deltas,2)
    rec_errors = np.log10(deltas_sq)
    return rec_errors
###############################################


###############################################
# Read in the data in a convenient form
def extractVariables(inputFile,maxEvents):
    dataContainer = {}
    featureNames = []
    lineCounter = -1
    inputFile.seek(0,0)
    nEntriesInLine = 0
    nNamedFeatures = 0
    for line in inputFile:
        lineCounter = lineCounter+1
        if (lineCounter % 100 == 0):
            sys.stdout.write("Reading %s: %d%%   \r" % (inputFile.name,100*lineCounter/maxEvents) )
            sys.stdout.flush()
        if lineCounter > maxEvents:
            break
        splitLine = line.split(",")
        nEntriesInLine = len(splitLine)
        if (lineCounter==0):
            nNamedFeatures = nEntriesInLine
            for item in splitLine:
                feature = item.strip('\n').strip("\r")
                featureNames.append(feature)
                dataContainer[feature] = []
        if (lineCounter>0):
            if nNamedFeatures!=nEntriesInLine:
                sys.stdout.write("\n")
                print "ERROR at event ",lineCounter
                sys.exit("Event contains more entries than the number of named features")
            for featureCounter in range(0,nNamedFeatures):
                key = featureNames[featureCounter]
                string = splitLine[featureCounter].strip('\n').strip('\r')
                try:
                    datum = float(string)
                    dataContainer[key].append(datum)
                except:
                    dataContainer[key].append(string)
    for key in dataContainer:
        dataContainer[key] = np.asarray(dataContainer[key])

    sys.stdout.write("\n")
    return dataContainer

###############################################

###############################################
# Build data arrays
def buildArrays(allowedFeatures,cut,data,skipEvents,nEvents,name):
    cut = cut[skipEvents:skipEvents+nEvents]
    outputArray = np.array([])
    for feature in data.keys():
        if feature in allowedFeatures:
            column = data[feature][skipEvents:skipEvents+nEvents]
            feature_vector = np.extract(cut,column)
            feature_vector = feature_vector.reshape(feature_vector.size,1)
            if outputArray.shape[0]==0:
                outputArray = feature_vector
            else:
                outputArray = np.concatenate((outputArray,feature_vector),axis=1)
    imp = Imputer(missing_values=-999, strategy='mean', axis=0)
    imp.fit(outputArray)
    outputArray = imp.transform(outputArray)
    print name
    print "Events: ",outputArray.shape[0]
    print "Features: ",outputArray.shape[1]
    return outputArray
###############################################

###############################################
# Generate ROC, precision,recall arrays
def makeMetrics(nPoints,anomalous,normal):
    mins = np.array([np.amin(anomalous),np.amin(normal)])
    maxs = np.array([np.amax(anomalous),np.amax(normal)])
    min = np.floor(np.amin(mins))
    max = np.ceil(np.amax(maxs))
    gradation = (max - min)/np.float(nPoints)
    true_positive_fractions = []
    false_positive_fractions = []
    precisions = []
    recalls = []
    distances = []
    f1s = []
    for slide in range(0,nPoints,1):
        cut = min + (slide*gradation)
        n_TP = float(np.sum(anomalous > cut))
        n_FP = float(np.sum(normal > cut))
        n_TN = float(np.sum(normal < cut))
        n_FN = float(np.sum(anomalous < cut))
        if (n_TP==0.0):
            break
        n_anomalous = float(anomalous.shape[0])
        n_normal = float(normal.shape[0])
        true_positive_fractions.append(n_TP/n_anomalous)
        false_positive_fractions.append(n_FP/n_normal)
        precision = n_TP/(n_TP+n_FP)
        recall = n_TP/(n_TP+n_FN)
        f1 = 2 * (precision*recall)/(precision+recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return np.asarray(true_positive_fractions),np.asarray(false_positive_fractions),np.asarray(precisions),np.asarray(recalls),np.asarray(f1s)

###############################################
# Area under curve
def areaUnderROC(true_positive_fractions,false_positive_fractions):
    X = np.sort(false_positive_fractions)
    Y = true_positive_fractions[np.argsort(false_positive_fractions)]
    area = np.trapz(Y,x=X)
    return area



