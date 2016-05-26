# susy-neuralnetworks
Studies on use of neural networks for separating supersymmetric particle decays from Standard Model backgrounds

## Pre-requisites
(1) Python
(2) Numpy (linear algebra), Matplotlib (graphics), SciKit-Learn (machine learning tool-kit): all available via a single installation of Anaconda - https://www.continuum.io
(3) SKNN (interface between SciKit-Learn and Lasagne/Theano enabling quick building of neural networks): http://scikit-neuralnetwork.readthedocs.io/en/latest/index.html

## Input data format
Currently the examples use CSV files, with the first row of each file containing the name of the variables, e.g.

``Var1Name,Var2Name,...,VarMName``
``Event1Val1,Event1Val2,...,Event1ValM``
``...``
``EventNVal1,EventNVal2,...,EventNValM``

**To do:** set up a converter to enable ROOT files to be read in directly

The assumption in the examples is that the background file is called ``SUSY_signal.csv`` and the background ``SM_background.csv``, but of course this can be changed in the scripts.

## Description of the python files
- ``CommonTools.py``: methods for
  - reading in the CSV files and converting the data into numpy arrays 
  - calculating the reconstruction error from the autoencoder predictions, the resulting ROC/precision-recall values, and the area under the ROC curve
- ``featuresLists.py``: contains lists of variables that should be included in the training/testing samples. Any variables in the CSV files that are not named in these lists will be omitted. 
- ``autoencoder.py, classifier.py``: main scripts for running training/testing/plotting, for a single set of hyperparameters, of the autoencoder/classifier (see more later)
- ``CrossValidation.py``: script running multiple training cycles with a variety of hyperparameters, for the purposes of cross-validation. Trained networks are written to pickle files.
- ``CrossValidation_results.py``: script testing the networks written by ``CrossValidation.py`` on independent data, enabling optimal choice of hyperparameters

## Models used
Both the autoencoder and the classifier are fully connected multilayer perceptrons, using a single hidden layer and with recified linear activation unit. 

### Autoencoder
An autoencoder (or more specifically a Diabalo Network) has two features:
- in the training the output targets are set identically equal to the inputs, such that the network learns the identity function 
- there are fewer units in the hidden layer than inputs/outputs, such that the network is forced to perform dimensionality reduction, similar (identical?) to principal components analysis
The idea is that a new event, taken from the same distribution as the training events, is fed into the network, the autoencoder will do a reasonable job of reconstructing it since it has learned the identity function. The difference between the input and output (reconstruction error) should therefore be small. Whereas, if the new event comes from some other distribution, the network will not have learned/compressed its features and the reconstruction error will be large. The aim is to build an autoencoder that can distinguish between signal and background events, having only been taught the background - so it is a one-class classifier - with the discriminating variable being the size of the reconstruction error provided by the trained autoencoder.

### Classifier
In training, the classifier used in the examples sets the targets to 1 for signal events, and 0 for backgrounds. In testing, if the output is below 0.5 the event is taken to be predicted to be background; above 0.5, signal. 

## Measuring performance
In a binary sample subject to a classifier there are four possible outcomes:
- signal is identified as signal (SS)
- signal is identified as background (SB)
- background is identified as signal (BS)
- background is identified as background (BB)

As the cut on the discriminating variable (reconstruction error or NN prediction) is modified, the number of events in each of these categories varies. Two fractions are of particular interest: SS/totalS (correctly identified signal fraction) and BS/totalB (wrongly identified backrgound fraction). For each value of the discriminating variable cut, these fractions will vary as events shuffle from one side of the cut to the other. If one plots these fractions against each other, for all values of the cut, a curve (the Receiver Operating Characteristic) will be formed, with the area under the curve (AUC) being a measure of the performance of the classification algorithm. An algorithm for which the background and the signal produce very similar distributions in the discriminating variable (e.g. no discriminating power; equal probability of getting the right or wrong answer) will have a ROC curve that describes a straight Y=X line, with the AUC being 0.5. A perfect classifier, with the distributions of the discriminating variable from signal/background being completely separated, will form a line Y=1 with AUC being 1.0. If the classifier was doing worse than random, then AUC would be below 0.5.

### Hyperparameters
It is instructive to play with the neural network hyperparameters (and tune with the cross validation scripts). Here are some such variables and observations:
- *Number of hidden layers*: default is one; increasing the number of layers slightly worsens the performance and significantly increases the training time. 
- *Number of units in the hidden layer*: for the autoencoder the best results seem to be obtained when there are fewer units than inputs (e.g. 10 for a sample with 36 input variables). Tuned via cross-validation (see below)
- *Activation function*: rectified linear unit used at all times; no attempt to modify it
- *Learning rule*: this is the method by which the cost function, generated by back-propagation of the network's errors, is minimized. Stocastic gradient descent without momentum, weight decay or dropout is used throughout (no attempt to modify). *To do:* study effects of modifying the learning rule.
- *Learning rate*: this is the rate at which the weights are incremented by the learning rule, with a smaller increment leading to more precise minimization but slower convergence. Tuned via cross-validation (see below) 
- *Number of training cycles*: this is the number of minimization cycles performed - tuned by cross validation (see below)
- *Batch size*: this is the number of events fed into the network between each run of minimization. By default it is 1, which is slower but has a lower memory footprint. Increasing the batch size leads to faster training but higher memory footprint (because the weight vectors are bigger). No obvious sign of an impact on the performance.

### Cross validation
