In main.py the functions of all individual files are called.

#  Data Preparation
- read_CSV reads the csv files and adds a type attribute

- check for any null values using pandas 

-> no null values encountered

- for each column in the dataset and for each wine type plot bar plots for the distribution of the features over the quality

-> red wine better with less volatile acidity, white wine better with less dioxides

-> other features normally distributed over quality

- visualize distribution of each features with histograms

-> all normally distributed

-> data cleaning not necessary

- compare mean and median of each feature for red and white wine

-> similar: citric acid, density, pH, sulphates, alcohol, quality

-> not similar: residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide

-> use features that are not similar for classification

- correlation matrix (absolute value)
    - red wine: 2/0, 2/1, 6/5, 7/0, 8/0, 8/2 are correlated
    - white wine: 6/5, 7/6, 7/3, 10/7 are correlated

- split up the data: 80% train, 10% test and 10% validation set (randomized)

# Model
## Classification Task
- trained logistic regression and support vector machine model
- extended with outlier removal, standardization and normalization (in this order)
-> test score increased after each modification
-> used last version with standardization, normalization and removed outliers

## Regression Task
- Multi Layer Perceptron in Pytorch
- KNN Regressor 
-> rounded predictions yield better results (mean of abs values for scoring)
-> best result with original raw data


# Tuning
## Classification Task
- already on 99% accuracy most of the time

## Regression Task
- iteratively adding the features that yield the best scores, also removes correlated/useless features
- then choosing nonlinear functions that transform the features based on scores (validation dataset)
- cross validation for hyper parameter tuning (both the kNN and MLP model)


# Parallelization
- run: "python main.py --parallel" for parallel execution, emit the flag for standard execution
- improvements: from ~160 seconds to about 85 seconds (on my machine)
- time measured with time.time() in python
- parallelized the whole tuning process and with a parallel backend (multithreading)
- additionally added more jobs to the cross validation

# Debugging
- Violin plot for visualizing quality of validation set and prediction models
- Barplot of 4 different metrics 
- Confusion matrix validation against predictions