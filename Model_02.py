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
    return logReg.predict(X_test)

# Support Vector Machines
def svclassifier(X_train, Y_train, X_test, Y_test):
    svm = SVC(decision_function_shape='ovo')
    svm.fit(X_train, Y_train)
    print("[SUPPORT VECTOR MACHINE]")
    print('test score: ',svm.score(X_test, Y_test))
    return svm.predict(X_test)


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(12, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
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
    trainloader = torch.utils.data.DataLoader(data_array, batch_size=46, shuffle=True, num_workers=1)
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        
        # Print epoch
        #print(f'Starting epoch {epoch+1}')
        
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
            '''
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 10))
                current_loss = 0.0
            '''

    # Process is complete.
    #print('Training process has finished.')
    #mlp.eval()

    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()
    Y_test = torch.from_numpy(Y_test)
    Y_test = Y_test.float()


    y_pred = mlp(X_test)
    y_pred_rounded = torch.round(y_pred)
    
    criterion = torch.nn.MSELoss()


    y_pred_round = torch.round(y_pred)

    mlpScore1 = torch.mean(torch.abs(y_pred - Y_test)).item()
    mlpScore2 = torch.mean(torch.abs(y_pred_round - Y_test)).item()
    print("[MLP REGRESSION MSE]")
    print("test score: " , mlpScore1, " (not rounded)")
    print("test score: " , mlpScore2, " (rounded)")
    

    return y_pred, y_pred_round

def knnRegression(X_train, Y_train, X_test, Y_test):
    print("[KNN REGRESSION]")
    knnRegressor = KNeighborsRegressor(n_neighbors=20, weights="distance")
    knnRegressor.fit(X_train, Y_train)
    y_pred = knnRegressor.predict(X_test)
    y_pred_round = np.round(y_pred)

    kNNscore1 = np.mean(np.abs(y_pred - Y_test))
    kNNscore2 = np.mean(np.abs(y_pred_round - Y_test))
    print("test score: " , kNNscore1, " (not rounded)")
    print("test score: " , kNNscore2, " (rounded)")
    return y_pred, y_pred_round


def basic_models(X_train, Y_train, X_test, Y_test, X_validation, Y_validation):


    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = removeOutliers(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)

    train_type, train_quality = Y_train[:,1], Y_train[:,0]
    test_type, test_quality = Y_test[:,1], Y_test[:,0]
    validation_type, validation_quality = Y_validation[:,1], Y_validation[:,0]

    X_train_regression = np.concatenate([X_train, train_type.reshape(train_type.shape[0],1)], axis=1)
    X_test_regression = np.concatenate([X_test, test_type.reshape(test_type.shape[0],1)], axis=1)

    # MODEL
    logReg(X_train, train_type, X_test, test_type)
    svclassifier(X_train, train_type, X_test, test_type)
    X_train_regression = np.concatenate([X_train, train_type.reshape(train_type.shape[0],1)], axis=1)
    X_test_regression = np.concatenate([X_test, test_type.reshape(test_type.shape[0],1)], axis=1)

    y_pred, y_pred_reg_mlp = MLPregression(X_train_regression, train_quality.reshape((train_quality.shape[0], 1)), X_test_regression, test_quality)
    y_pred, y_pred_reg_knn = knnRegression(X_train_regression, train_quality, X_test_regression, test_quality)

    # standardization
    X_train, X_test, X_validation = scaler(X_train, X_test, X_validation)
    logReg(X_train, train_type, X_test, test_type)
    svclassifier(X_train, train_type, X_test, test_type)
    X_train_regression = np.concatenate([X_train, train_type.reshape(train_type.shape[0],1)], axis=1)
    X_test_regression = np.concatenate([X_test, test_type.reshape(test_type.shape[0],1)], axis=1)
    y_pred = MLPregression(X_train_regression, train_quality.reshape((train_quality.shape[0], 1)), X_test_regression, test_quality)
    y_pred = knnRegression(X_train_regression, train_quality, X_test_regression, test_quality)

    # normalization
    X_train, X_test, X_validation = normalize(X_train, X_test, X_validation)
    y_pred_class_logreg = logReg(X_train, train_type, X_test, test_type)
    y_pred_class_svm = svclassifier(X_train, train_type, X_test, test_type)
    
    X_train_regression = np.concatenate([X_train, train_type.reshape(train_type.shape[0],1)], axis=1)
    X_test_regression = np.concatenate([X_test, test_type.reshape(test_type.shape[0],1)], axis=1)
    
    MLPregression(X_train_regression, train_quality.reshape((train_quality.shape[0], 1)), X_test_regression, test_quality)
    knnRegression(X_train_regression, train_quality, X_test_regression, test_quality)

    
    return y_pred_class_logreg, y_pred_class_svm, y_pred_reg_mlp, y_pred_reg_knn

