import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import random as rnd

import sklearn.metrics as metrics
import sklearn.utils

from sklearn.neural_network import MLPRegressor


def scaler(train, test, valid):
    standardScaler = StandardScaler()
    std = standardScaler.fit(train)
    trainscaled = std.transform(train)
    testscaled = std.transform(test)
    validscaled = std.transform(valid)
    return trainscaled, testscaled, validscaled

def normalize(train, test, valid):
    normalizer = MinMaxScaler()
    minmax = normalizer.fit(train)
    trainnorm = minmax.transform(train)
    testnorm = minmax.transform(test)
    validnorm = minmax.transform(valid)
    return trainnorm, testnorm, validnorm

def removeOutliers(X_train, Y_train,  X_test, Y_test, X_validation, Y_validation):
    iso = IsolationForest(contamination=0.1)
    maskTrain = iso.fit_predict(X_train) != -1
    maskTest = iso.predict(X_test) != -1
    maskValid = iso.predict(X_validation) != -1
    return X_train[maskTrain], Y_train[maskTrain], X_test[maskTest], Y_test[maskTest], X_validation[maskValid], Y_validation[maskValid]

# Logistic Regression
def logReg(X_train, Y_train, X_test, Y_test):
    logReg = LogisticRegression(max_iter=10000)
    logReg.fit(X_train, Y_train)
    predictedTypes = logReg.predict(X_test)
    print("[LOGISTIC REGRESSION]")
    print('test score: ',logReg.score(X_test, Y_test))
    return logReg

# Support Vector Machines
def svclassifier(X_train, Y_train, X_test, Y_test):
    svm = SVC(decision_function_shape='ovo')
    svm.fit(X_train, Y_train)
    print("[SUPPORT VECTOR MACHINE]")
    print('test score: ',svm.score(X_test, Y_test))
    return logReg

# MLP regression
def MLPreg(X_train, Y_train, X_test, Y_test):
    parameters = [{  # 'hidden_layer_sizes': [3, 5, 10, 100],
        'alpha': [0.01, 0.03, 0.1],
        'activation': ['relu', 'identity']}]
    # 'activation': ['relu','logistic','tanh']}]

    regressor = MLPRegressor(solver="adam", max_iter=10000, activation="relu", alpha=0.01,
                             hidden_layer_sizes=(24, 30, 24))
    # regressor = GridSearchCV(regressor, parameters, verbose=3)
    regressor.fit(X_train, Y_train)
    # best_parameters = regressor.best_params_

    # print ('best parameters:', best_parameters)

    # best parameters: {'activation': 'logistic', 'alpha': 1, 'hidden_layer_sizes': 100}

    regressorResult = regressor.predict(X_test)
    for i in range(0, len(regressorResult)):
        regressorResult[i] = np.round(regressorResult[i])
    #plt.close("all")
    #plt.matshow(metrics.confusion_matrix(Y_test, regressorResult))
    #plt.show()

    trainResult = regressor.predict(X_train)
    regressorScore = np.mean(np.abs(trainResult - Y_train))

    print("Average Distance train: ", regressorScore)

    regressorScore = np.mean(np.abs(regressorResult - Y_test))

    print("Average Distance: ", regressorScore)
    return regressor


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(11, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

def MLPregression(X_train, Y_train, X_test, Y_test ):
    data_array = np.concatenate([X_train, Y_train], axis=1)
    data_array = torch.from_numpy(data_array)
    trainloader = torch.utils.data.DataLoader(data_array, batch_size=10, shuffle=True, num_workers=1)
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
        
            # Get and prepare inputs
            targets = torch.from_numpy(Y_train)
            inputs = torch.from_numpy(X_train)
            inputs, targets = inputs.float(), targets.float()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')