import csv
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



file_red = open("winequality-red.csv")
file_white = open("winequality-white.csv")


csvreader = csv.reader(file_red)

header = next(csvreader)
print(header)


rows = []
for row in csvreader:
        newRow = []
        newRow = str(row[0]).split(";");
        if(int(newRow[11]) < 0 or int(newRow[11]) > 10):
            print("ALARM");
        
        newRow.append(1)
        rows.append(newRow)

len_white = len(rows)

csvreader = csv.reader(file_white)
header = next(csvreader)
print(header)
for row in csvreader:
        newRow = str(row[0]).split(";")
        if(int(newRow[11]) < 0 or int(newRow[11]) > 10):
            print("ALARM")
        newRow.append(0)
        rows.append(newRow)

header = str(header[0]).split(";")
print(header)








new_header = []

for h in header:
    h = h.replace('"', '')
    new_header.append(h)

for i in new_header:
    plt.figure()
    plt.title("Red Wine Bar Plot with " + i)
    plt.bar(wine_data_red["quality"], wine_data_red[i])
    plt.savefig("figures/bar_red"+i)
    plt.close()

    plt.figure()
    plt.title("White Wine Bar Plot with " + i)
    plt.bar(wine_data_white["quality"], wine_data_white[i])
    plt.savefig("figures/bar_white"+i)
    plt.close()

    plt.figure()
    plt.title("Merged Wine Bar Plot with " + i)
    plt.bar(wine_data_merged["quality"], wine_data_merged[i])
    plt.savefig("figures/bar_merged"+i)
    plt.close()


len_red = len(rows) - len_white

rows_array = np.asarray(rows, dtype=np.float)

means = []
stds = []
medians = []

np.random.shuffle(rows_array)

cut_1 = int(0.8*np.shape(rows_array)[0])
cut_2 = int(0.9*np.shape(rows_array)[0])

training, test, verification = rows_array[:cut_1,:], rows_array[cut_1 + 1:cut_2,: ], rows_array[cut_2:, :]

print(np.shape(training), np.shape(test), np.shape(verification))

for i in range(0, len(header)):
    wine_feature = training[:,i]
    white_feature = wine_feature[0:len_white - 1]
    red_feature = wine_feature[len_white : ]

    plt.figure()
    n, bins, patches = plt.hist(red_feature, 40, facecolor='r')
    n, bins, patches = plt.hist(white_feature, 40, facecolor='g')
    plt.title(header[i])
    #plt.savefig("figures/"+str(i)+"histogram"+header[i])
    plt.close()

    mean_white = np.mean(wine_feature[0:len_white - 1])
    mean_red = np.mean(wine_feature[len_white :])
    means.append((mean_red, mean_white))


    std_white = np.std(wine_feature[0:len_white - 1])
    std_red = np.std(wine_feature[len_white:])
    stds.append((mean_red, mean_white))

    median_white = np.median(wine_feature[0:len_white - 1])
    median_red = np.median(wine_feature[len_white:])
    medians.append((median_red, median_white))

    print(header[i])
    print("MEAN RED - MEAN WHITE")
    print(mean_red, mean_white)
    print("STD RED - STD WHITE")
    print(std_red, std_white)
    print("MEDIAN RED - MEDIAN WHITE")
    print(median_red, median_white)
    print()

train_red = []
train_white = []


assert(False)





'''

for i in range(0, len(header)):
    for j in range(0, len(header)):

        if i == j:
            continue
        wine_feature1 = rows_array[:,i]
        wine_feature2 = rows_array[:,j]

        white_feature1 = wine_feature1[0:len_white - 1]
        white_feature2 = wine_feature2[0:len_white - 1]

        red_feature1 = wine_feature1[len_white : ]
        red_feature2 = wine_feature2[len_white : ]

        plt.figure()
        plt.scatter(red_feature1, red_feature2, c="r", marker="x")
        plt.scatter(white_feature1, white_feature2, c="g", alpha=0.2)
        plt.title(header[i] + header[j])
        plt.savefig("figures/scatter"+header[i]+header[j])
'''

R2 = np.corrcoef(training.T)
plt.matshow(np.abs(R2))
plt.show()
R2 = R2[-1]
'''
resultsum = np.zeros(test[:,-1].shape)
for j in range(0, len(header)):
    classification = []
    classification1 = []
    for i in range (0, test.shape[0]):
        prob_red = scipy.stats.norm(means[j][0], stds[j][0]).pdf(test[i][j])
        prob_white = scipy.stats.norm(means[j][1], stds[j][1]).pdf(test[i][j])
        if prob_red > prob_white:
            classification.append(-1)
            classification1.append(0)
        else:
            classification.append(1)
            classification1.append(1)

        scipy.stats.norm(0, 1).pdf(0)
    result = np.mean(np.abs(classification1 - test[:,-1]))
    print(header[j])
    print(result)
    print(R2[j])

    resultsum += abs(R2[j]) * np.array(classification)
resultsum = resultsum > 0
result = np.mean(np.abs(resultsum - test[:, -1]))
print(result)
    #print(classification)
    #print(rows_array[:,-1])
'''
svm = SVC(decision_function_shape='ovo')
svm.fit(training[:,:-1], training[:,-1])
result = svm.predict(test[:,:-1])
print(np.mean(np.abs(result-test[:,-1])))

'''
training = np.delete(training, 6, 1)
test = np.delete(test, 6, 1)
svm = SVC(decision_function_shape='ovo')
svm.fit(training[:,:-1], training[:,-1])
result = svm.predict(test[:,:-1])
print(np.mean(np.abs(result-test[:,-1])))
'''


logre = LogisticRegression(max_iter=10000)
logre.fit(training[:,:-1], training[:,-1])
trainres = logre.predict(training[:,:-1])
print(np.mean(np.abs(trainres-training[:,-1])))
res = logre.predict(test[:,:-1])
print(np.mean(np.abs(res-test[:,-1])))
valres = logre.predict(verification[:,:-1])
print(np.mean(np.abs(valres-verification[:,-1])))
'''
R2 = np.corrcoef(rows_array.T)

print(R2)
'''
