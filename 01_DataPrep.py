import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    # Bar plots with features over quality
    '''
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
    '''

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

    '''
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

X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=1)
X_test, X_validation, Y_test, Y_validation = ms.train_test_split(X_test, Y_test, test_size=0.5, random_state=1)

Y_train_quality = Y_train.T[0].T
Y_train_type = Y_train.T[1].T

Y_test_quality = Y_test.T[0].T
Y_test_type = Y_test.T[1].T

Y_validation_quality = Y_validation.T[0].T
Y_validation_type = Y_validation.T[1].T


# outlier removal
'''
iso = IsolationForest(contamination=0.5)
mask = iso.fit_predict(X_train)
mask = mask != -1
X_train, Y_train_type = X_train[mask, :], Y_train_type[mask]
'''

# normalization DOES NOT WORK WELL
'''
minmaxScaler = MinMaxScaler()
norm = minmaxScaler.fit(X_train)
X_train = norm.transform(X_train)
'''

# standardization
'''
standardScaler = StandardScaler()
std = standardScaler.fit(X_train)
X_train = std.transform(X_train)
print(X_train)
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



