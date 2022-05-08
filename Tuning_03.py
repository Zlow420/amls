import numpy as np
from Model_02 import knnRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

nonlins = [np.sin, np.square, np.sqrt, np.tanh, np.exp, ]


def add_nonlin(dataset, index, nonlin):
    feature = dataset[:,index]
    nonlin_feat = nonlin(feature)
    new_data = np.concatenate([dataset, nonlin_feat.reshape((nonlin_feat.shape[0], 1))], 1)
    return new_data

#todo: try composite nonlinearities
#def add_composite_nonlin(dataset, index, nonlin):


#todo: test for removal of original feature
def try_nonlins(X_train, Y_train, X_validation, Y_validation):
    train_type, train_quality = Y_train[:, 1], Y_train[:, 0]
    validation_type, validation_quality = Y_validation[:, 1], Y_validation[:, 0]
    X_train_regression = np.concatenate([X_train, train_type.reshape(train_type.shape[0], 1)], axis=1)
    X_valid_regression = np.concatenate([X_validation, validation_type.reshape(validation_type.shape[0], 1)], axis=1)
    _,_,bestscore = knnRegression(X_train_regression, train_quality, X_valid_regression, validation_quality)
    accepted_nonlins = []
    for i in range(X_train.shape[1]):
        for nl in nonlins:
            new_X_train = add_nonlin(X_train_regression, i, nl)
            new_X_valid = add_nonlin(X_valid_regression, i, nl)
            _,_,nonlinscore = knnRegression(new_X_train, train_quality, new_X_valid, validation_quality)
            if nonlinscore < bestscore:
                print('accepted new nonlinearity: ', i, nl, nonlinscore)
                bestscore = nonlinscore
                accepted_nonlins.append((i, nl))
                X_train_regression = new_X_train
                X_valid_regression = new_X_valid


    return X_train_regression, X_valid_regression



def add_features(X_train, Y_train, X_validation, Y_validation, numberOfFeatures):
    train_type, train_quality = Y_train[:, 1], Y_train[:, 0]
    validation_type, validation_quality = Y_validation[:, 1], Y_validation[:, 0]
    X_train_regression = np.concatenate([X_train, train_type.reshape(train_type.shape[0], 1)], axis=1)
    X_valid_regression = np.concatenate([X_validation, validation_type.reshape(validation_type.shape[0], 1)], axis=1)

    best_indices = []
    ret_x_train = np.zeros(1)
    ret_x_valid = np.zeros(1)
    
    for iteration in range(0, numberOfFeatures):
        new_X_train = np.zeros((X_train_regression.shape[0], iteration))
        new_X_valid = np.zeros((X_valid_regression.shape[0], iteration))
        current_best = 999999 
        current_best_index = 0

        for index in range(len(best_indices)):
            new_X_train[:,index] = X_train_regression[:,index]
            new_X_valid[:,index] = X_valid_regression[:,index]


        for i in range(X_train_regression.shape[1]):
            X_train_regression_feature = np.array(X_train_regression[:,i])
            X_valid_regression_feature = np.array(X_valid_regression[:,i])

            tmp_X_train = np.concatenate([new_X_train, X_train_regression_feature.reshape(X_train_regression.shape[0],1)], axis=1)
            tmp_X_valid = np.concatenate([new_X_valid, X_valid_regression_feature.reshape(X_valid_regression.shape[0],1)], axis=1)
            _,_,nonlinscore = knnRegression(tmp_X_train, train_quality, tmp_X_valid, validation_quality)

            
            if nonlinscore < current_best and i not in best_indices:
               print("NEW SCORE:", nonlinscore)
               current_best = nonlinscore
               current_best_index = i
               ret_x_train = tmp_X_train
               ret_x_valid = tmp_X_valid

        best_indices.append(current_best_index)
    
    return best_indices, ret_x_train, ret_x_valid


def perform_tuning(X_train, Y_train, X_validation, Y_validation, parallel=False):
    if parallel:
        numberOfJobs = 1
    else:
        numberOfJobs = 2

    print("ADDING SINGLE FEATURES")
    best_indices, new_X_train, new_X_valid = add_features(X_train, Y_train, X_validation, Y_validation, 4)

    print("ADDING NONLINEARITIES")
    new_X_train, new_X_valid = try_nonlins(new_X_train, Y_train, new_X_valid, Y_validation)

    train_type, train_quality = Y_train[:,1], Y_train[:,0]
    validation_type, validation_quality = Y_validation[:,1], Y_validation[:,0]


    #discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    #discretizer.fit(new_X_train)    

    #new_X_train = discretizer.transform(new_X_train)
    #new_X_valid = discretizer.transform(new_X_valid)

    print("CROSS VALIDATION FOR KNN")
    knnRegressor = KNeighborsRegressor()

    params = [{'knn__n_neighbors': [5,10,15,20,25,30,40,50,60,70,80,90,100],
         'knn__weights': ['uniform', 'distance'],
         "pca__n_components": [1,2,3,4,5,6],
         }]

    scaler = StandardScaler()
    minmax = MinMaxScaler()
    pca = PCA()
    pipe = Pipeline(steps=[("scaler", scaler), ("minmax", minmax), ("pca", pca), ("knn", knnRegressor)])

    gs_knn = GridSearchCV(pipe,
                      param_grid=params,
                      verbose=3, cv=10, n_jobs = numberOfJobs)

    gs_knn.fit(new_X_train, train_quality)
    kNNscore = np.mean(np.abs(gs_knn.predict(new_X_valid) - validation_quality))    

    print("NEW KNN SCORE: " , kNNscore)
    print("best params:", gs_knn.best_params_)


    kNNresult = gs_knn.predict(new_X_valid)
    kNNresult = np.round(kNNresult)


    print("CROSS VALIDATION FOR MLP")

    params = [{
        #"mlp__hidden_layer_sizes" : [(256,), (128,), (64,), (32,)],
        "mlp__hidden_layer_sizes" : [(64,), (32,)],
        #"mlp__activation" : ["identity","logistic", "relu"],
        "mlp__activation" : ["identity", "relu"],
        "mlp__alpha" : [0.0001, 0.1],
        #"mlp__alpha" : [0.0001, 0.001, 0.01, 0.1],
        #"mlp__batch_size": [10, 50, 100],
        "pca__n_components": [2,5,6],
        }]

    mlpRegressor = MLPRegressor(activation="relu", hidden_layer_sizes=(64,))

    '''

    mlpRegressor = MLPRegressor(activation="relu")

    pipe = Pipeline(steps=[("scaler", scaler), ("minmax", minmax), ("pca", pca), ("mlp", mlpRegressor)])


    gs_mlp = GridSearchCV(pipe,
                      param_grid=params,
                      verbose=3, cv=10, n_jobs = numberOfJobs)

    gs_mlp.fit(new_X_train, train_quality)
    kNNscore = np.mean(np.abs(gs_mlp.predict(new_X_valid) - validation_quality))    


    print("NEW KNN SCORE: " , kNNscore)
    print("best params:", gs_mlp.best_params_)


    MLPresult = gs_knn.predict(new_X_valid)
    MLPresult = np.round(MLPresult)
    '''

    mlpRegressor.fit(new_X_train, train_quality)
    MLPresult = mlpRegressor.predict(new_X_valid)

    MLPresult = np.round(MLPresult)


    #return kNNresult, MLPresult, gs_knn, gs_mlp
    return kNNresult, MLPresult, gs_knn, mlpRegressor

