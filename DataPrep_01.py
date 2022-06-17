import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection as ms
import random as rnd
import seaborn as sns

# Open CSV files for red and white whine, then merge them
def read_CSV():
    wine_data_white = pd.read_csv("winequality-white.csv", sep=";")
    wine_data_white["type"] = 1

    wine_data_red = pd.read_csv("winequality-red.csv", sep=";")
    wine_data_red["type"] = 0

    wine_files = [wine_data_red, wine_data_white]
    wine_data_merged = pd.concat(wine_files)

    # Display info about the files, check for null values
    wine_data_merged.info(verbose=True)
    wine_data_red.info(verbose=True)
    wine_data_white.info(verbose=True)

    # Retrieve column names
    columns = wine_data_merged.columns

    '''
    for c in columns:
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
    
        '''
    # Correlation Matrix
    corr_matrix_red = np.corrcoef(wine_data_red.drop(['type'], axis=1).T)
    corr_matrix_white = np.corrcoef(wine_data_white.drop(['type'], axis=1).T)

    plt.figure(figsize=(10,10))
    sns.heatmap(np.abs(corr_matrix_red), annot=True, xticklabels=columns.drop(['type']), yticklabels=columns.drop(['type']))
    plt.title("Correlation Matrix")
    plt.savefig("figures/[CORRELATION MATRIX RED]")
    plt.close()

    plt.figure(figsize=(10,10))
    sns.heatmap(np.abs(corr_matrix_white), annot=True, xticklabels=columns.drop(['type']), yticklabels=columns.drop(['type']))
    plt.title("Correlation Matrix")
    plt.savefig("figures/[CORRELATION MATRIX WHITE]")
    plt.close()

    # Split up to 80% Train, 10% Test, 10% Validation data
    wine_data_merged = np.array(wine_data_merged, dtype=float)

    X = wine_data_merged.T[:-2].T
    Y = wine_data_merged.T[-2:].T

    randomState = rnd.randint(0, 900000)

    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=randomState)
    X_test, X_validation, Y_test, Y_validation = ms.train_test_split(X_test, Y_test, test_size=0.5, random_state=randomState)

    Y_train_quality = Y_train[:,0]
    Y_train_type = Y_train[:,1]
    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation
