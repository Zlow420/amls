import numpy as np
from Model_02 import knnRegression


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



