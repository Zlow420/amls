import pandas as pd
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

from sklearn.neural_network import MLPRegressor

# Open CSV files for red and white whine, then merge them
wine_data_white = pd.read_csv("winequality-white.csv", sep=";")
wine_data_white["type"] = 1;

wine_data_red = pd.read_csv("winequality-red.csv", sep=";")
wine_data_red["type"] = 0;

wine_files = [wine_data_red, wine_data_white]
wine_data_merged = pd.concat(wine_files)

# Display info about the files, check for null values
wine_data_merged.info(verbose=True)
wine_data_red.info(verbose=True)
wine_data_white.info(verbose=True)

# Retrieve column names
columns = wine_data_merged.columns

for c in columns:
    '''
    # Bar plots with features over quality
    fig = plt.figure(figsize=(4,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    plt.subplots_adjust(hspace=0.4, wspace=0.6)

    ax1.set_title("Red Wine Bar Plot with " + c)
    ax1.bar(wine_data_red["quality"], wine_data_red[c])

    ax2.set_title("White Wine Bar Plot with " + c)
    ax2.bar(wine_data_white["quality"], wine_data_white[c])

    ax1.set(xlabel="Quality", ylabel=c)
    ax2.set(xlabel="Quality", ylabel=c)

    plt.savefig("figures/[BARPLOT] "+c)
    plt.close()
    

    # Histograms for each feature
    fig = plt.figure(figsize=(4,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    ax1.set_title("Red Wine Histogram with " + c)
    ax1.hist(wine_data_red[c], 40)

    ax2.set_title("White Wine Histogram with " + c)
    ax2.hist(wine_data_white[c], 40)

    ax1.set(xlabel=c)
    ax2.set(xlabel=c)

    plt.savefig("figures/[HISTOGRAM] "+c)
    plt.close()

    # Comparison between mean and median values for every feature
    white_mean = np.mean(wine_data_white[c])
    white_std = np.std(wine_data_white[c])
    white_median = np.median(wine_data_white[c])

    red_mean = np.mean(wine_data_red[c])
    red_std = np.std(wine_data_red[c])
    red_median = np.median(wine_data_red[c])

    print(c)
    print("WHITE WINE")
    print("Mean: ", white_mean, "Std: ", white_std, "Median: ", white_median)
    print("RED WINE")
    print("Mean: ", red_mean, "Std: ", red_std, "Median: ", red_median)
    print()

    # Scatter Plots to visualize correlation between features
    for c2 in columns:
        if c == c2:
            continue
        fig = plt.figure(figsize=(4,10))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax2)

        ax1.scatter(wine_data_red[c], wine_data_red[c2], c="r")
        ax1.set_title("Red Wine Data")
        ax2.scatter(wine_data_white[c], wine_data_white[c2], c="b")
        ax2.set_title("White Wine Data")
        plt.title(c + "-" + c2)
        plt.savefig("figures/[SCATTER] "+ c + "-" + c2)
        plt.close
'''
    # Correlation Matrix
    corr_matrix_red = np.corrcoef(wine_data_red.T)
    corr_matrix_white = np.corrcoef(wine_data_white.T)
    fig = plt.figure(figsize=(4,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.matshow(np.abs(corr_matrix_red))
    ax2.matshow(np.abs(corr_matrix_white))
    plt.title("Correlation Matrices")
    plt.savefig("figures/[CORRELATION MATRIX]")
    plt.close



# Split up to 80% Train, 10% Test, 10% Validation data
wine_data_merged_matrix = list(wine_data_merged)
wine_data_merged = np.array(wine_data_merged, dtype=float)

X = wine_data_merged.T[:-2].T
Y = wine_data_merged.T[-2:].T

randomState = rnd.randint(0, 900000)

X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=randomState)
X_test, X_validation, Y_test, Y_validation = ms.train_test_split(X_test, Y_test, test_size=0.5, random_state=randomState)

Y_train_quality = Y_train.T[0].T
Y_train_type = Y_train.T[1].T

Y_train_quality_binary = []
for i in range(0, len(Y_train_quality)):
    if(Y_train_quality[i] <= 4):
        Y_train_quality_binary.append(0)
    else:
        Y_train_quality_binary.append(1)


Y_test_quality = Y_test.T[0].T
Y_test_type = Y_test.T[1].T

Y_test_quality_binary = []
for i in range(0, len(Y_test_quality)):
    if(Y_test_quality[i] <= 4):
        Y_test_quality_binary.append(0)
    else:
        Y_test_quality_binary.append(1)




Y_validation_quality = Y_validation.T[0].T
Y_validation_type = Y_validation.T[1].T


# outlier removal

'''
iso = IsolationForest(contamination=0.0001)
mask = iso.fit_predict(X_train)
mask = mask != -1
X_train, Y_train_type, Y_train_quality = X_train[mask, :], Y_train_type[mask], Y_train_quality[mask]
'''


# normalization DOES NOT WORK WELL
'''
minmaxScaler = MinMaxScaler()
norm = minmaxScaler.fit(X_train)
X_train = norm.transform(X_train)
'''



# Logistic Regression
logReg = LogisticRegression(max_iter=10000)
logReg.fit(X_train, Y_train_type)
predictedTypes = logReg.predict(X_test)
print("[LOGISTIC REGRESSION]")
print(logReg.score(X_test, Y_test_type))

# Support Vector Machines
svm = SVC(decision_function_shape='ovo')
svm.fit(X_train, Y_train_type)
print("[SUPPORT VECTOR MACHINE]")
print(svm.score(X_test, Y_test_type))

# use wine type for quality prediction
X_train = np.concatenate([X_train, Y_train_type.reshape(Y_train_type.shape[0],1)], axis=1)
X_test = np.concatenate([X_test, Y_test_type.reshape(Y_test_type.shape[0],1)], axis=1)


num_neighbors = 20

# standardization
standardScaler = StandardScaler()
std = standardScaler.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)
print(X_train)

normalizer = MinMaxScaler()
minmax = normalizer.fit(X_train)
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)

# creating artificial feature -> low/high quality of wine
toConcatetate = [6, 4, 1]
new_X_train_class = np.concatenate([X_train.T[5].reshape(len(X_train.T[5]),1), Y_train_type.reshape(len(Y_train_type),1)], axis=1)
new_X_test_class = np.concatenate([X_test.T[5].reshape(X_test.shape[0],1), Y_test_type.reshape(len(Y_test_type),1)], axis=1)

for i in toConcatetate:
    new_X_train_class = np.concatenate([X_train.T[i].reshape(len(X_train.T[i]),1), new_X_train_class], axis=1)
    new_X_test_class = np.concatenate([X_test.T[i].reshape(len(X_test.T[i]),1), new_X_test_class], axis=1)

print("[KNN Classification] determining high/low quality")
kNNclassifier = KNeighborsClassifier(n_neighbors=num_neighbors, weights="distance")
kNNclassifier.fit(new_X_train_class, Y_train_quality_binary)
X_test_quality_binary = kNNclassifier.predict(new_X_test_class)
kNNscore = kNNclassifier.score(new_X_test_class, Y_test_quality_binary)
print("Score: " , kNNscore)

Y_train_quality_binary = np.array(Y_train_quality_binary)
X_test_quality_binary = np.array(X_test_quality_binary)

#taking features one by one to increase accuracy
#free sulfur oxide
new_X_train = np.concatenate([X_train.T[5].reshape(len(X_train.T[5]),1), Y_train_type.reshape(len(Y_train_type),1)], axis=1)
new_X_test = np.concatenate([X_test.T[5].reshape(X_test.shape[0],1), Y_test_type.reshape(len(Y_test_type),1)], axis=1)

'''
new_X_train = np.concatenate([Y_train_quality_binary.reshape(len(X_train.T[6]),1), new_X_train], axis=1)
new_X_test = np.concatenate([X_test_quality_binary.reshape(len(X_test.T[6]),1), new_X_test], axis=1)
'''

def color(quality):
    if(quality == 3):
        return "red"
    elif(quality == 4):
        return "green"
    elif(quality == 5):
        return "blue"
    elif(quality == 6):
        return "yellow"
    elif(quality == 7):
        return "pink"
    elif(quality == 8):
        return "purple"
    elif(quality == 9):
        return "black"
    else: 
        assert(False)


plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')



#for i in range(0, len(X_train.T[5])):
#    ax.scatter(X_train.T[5][i], X_train.T[10][i], X_train.T[1][i], color=color(Y_train_quality.T[i]))
#plt.show()



print("[KNN REGRESSION] + free sulfur oxide + standardization")
kNNregressor = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
kNNregressor.fit(new_X_train, Y_train_quality)
kNNresult = kNNregressor.predict(new_X_test)
kNNscore = np.mean(np.abs(kNNresult - Y_test_quality))
print("Average Distance: " , kNNscore)

#total sulfur oxide
new_X_train = np.concatenate([X_train.T[6].reshape(len(X_train.T[6]),1), new_X_train], axis=1)
new_X_test = np.concatenate([X_test.T[6].reshape(len(X_test.T[6]),1), new_X_test], axis=1)

print("[KNN REGRESSION] + total sulfur oxide")
kNNregressor = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
kNNregressor.fit(new_X_train, Y_train_quality)
kNNresult = kNNregressor.predict(new_X_test)
kNNscore = np.mean(np.abs(kNNresult - Y_test_quality))
print("Average Distance: " , kNNscore)

#volatile acidity 
new_X_train = np.concatenate([X_train.T[7].reshape(len(X_train.T[7]),1), new_X_train], axis=1)
new_X_test = np.concatenate([X_test.T[7].reshape(len(X_test.T[7]),1), new_X_test], axis=1)

print("[KNN REGRESSION] + volatile acidity")
kNNregressor = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
kNNregressor.fit(new_X_train, Y_train_quality)
kNNresult = kNNregressor.predict(new_X_test)
kNNscore = np.mean(np.abs(kNNresult - Y_test_quality))
print("Average Distance: " , kNNscore)

#fixed acidity 
new_X_train = np.concatenate([X_train.T[0].reshape(len(X_train.T[0]),1), new_X_train], axis=1)
new_X_test = np.concatenate([X_test.T[0].reshape(len(X_test.T[0]),1), new_X_test], axis=1)

print("[KNN REGRESSION] + fixed acidity")
kNNregressor = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
kNNregressor.fit(new_X_train, Y_train_quality)
kNNresult = kNNregressor.predict(new_X_test)
kNNscore = np.mean(np.abs(kNNresult - Y_test_quality))
print("Average Distance: " , kNNscore)

#alcohol 
new_X_train = np.concatenate([X_train.T[10].reshape(len(X_train.T[10]),1), new_X_train], axis=1)
new_X_test = np.concatenate([X_test.T[10].reshape(len(X_test.T[10]),1), new_X_test], axis=1)

print("[KNN REGRESSION] + fixed acidity")
kNNregressor = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
kNNregressor.fit(new_X_train, Y_train_quality)
kNNresult = kNNregressor.predict(new_X_test)
kNNscore = np.mean(np.abs(kNNresult - Y_test_quality))
print("Average Distance: " , kNNscore)

parameters = [{'n_neighbors': [1, 5, 10, 20, 50, 100, 200],
                   'weights': ["uniform", "distance"],
                   }]

kNNregressor = KNeighborsRegressor()
kNNregressor = GridSearchCV(kNNregressor, parameters, verbose=3)
kNNregressor.fit(new_X_train, Y_train_quality)
best_parameters = kNNregressor.best_params_

print ('best parameters:', best_parameters)


print("[KNN REGRESSION] + rounding")
for i in range(0, len(kNNresult)):
    kNNresult[i] = np.round(kNNresult[i])
kNNscore = np.mean(np.abs(kNNresult - Y_test_quality))
print("Average Distance: " , kNNscore)



plt.close("all")
plt.matshow(metrics.confusion_matrix(Y_test_quality, kNNresult, labels=[3,4,5,6,7,8,9]))
plt.show()




parameters = [{'hidden_layer_sizes': [3, 5, 10, 100],
                   'alpha': [0.01, 1, 10, 100],
                   'activation': ['relu','logistic','tanh', 'identity']}]


regressor = MLPRegressor(solver="lbfgs", max_iter=10000, activation="logistic", alpha=1, hidden_layer_sizes=100)
#regressor = GridSearchCV(regressor, parameters, verbose=3)
regressor.fit(X_train, Y_train_quality)
#best_parameters = regressor.best_params_

#print ('best parameters:', best_parameters)

# best parameters: {'activation': 'logistic', 'alpha': 1, 'hidden_layer_sizes': 100}

regressorResult = regressor.predict(X_test)
for i in range(0, len(regressorResult)):
    regressorResult[i] = np.round(regressorResult[i])
plt.close("all")
plt.matshow(metrics.confusion_matrix(Y_test_quality, regressorResult))
plt.show()



regressorScore = np.mean(np.abs(regressorResult - Y_test_quality))

print("Average Distance: " , regressorScore)



