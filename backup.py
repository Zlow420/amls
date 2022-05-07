

def todo():
    Y_train_quality_binary = []
    for i in range(0, len(Y_train_quality)):
        if(Y_train_quality[i] <= 4):
            Y_train_quality_binary.append(0)
        else:
            Y_train_quality_binary.append(1)


    Y_test_quality = Y_test[:,0]
    Y_test_type = Y_test[:,1]

    Y_train_quality = Y_train[:,0]

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


    '''
    for i in range(0, len(X_train.T[5])):
        ax.scatter(X_train.T[5][i], X_train.T[10][i], X_train.T[1][i], color=color(Y_train_quality.T[i]))
    plt.show()
    '''


    print("[KNN REGRESSION] + free sulfur oxide")
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



    '''
    plt.close("all")
    plt.matshow(metrics.confusion_matrix(Y_test_quality, kNNresult, labels=[3,4,5,6,7,8,9]))
    plt.show()
    '''


    quality_split = []
    for q in range(3, 10):
        quality_split.append(X_train[Y_train_quality == q])

    cut_size = 50

    splitcut = cut_size if len(quality_split[0]) > cut_size else len(quality_split[0])
    sampled_train_X = quality_split[0][:splitcut]
    sampled_train_Y = np.ones_like(sampled_train_X.T[0].T)*3

    for i in range(1,len(quality_split)):
        splitcut = cut_size if len(quality_split[i]) > cut_size else len(quality_split[i])
        sampled_train_X_ = quality_split[i][:splitcut]
        sampled_train_Y_ = np.ones_like(sampled_train_X_.T[0].T) * (i+3)
        sampled_train_X = np.concatenate([sampled_train_X,sampled_train_X_])
        sampled_train_Y = np.concatenate([sampled_train_Y,sampled_train_Y_])

    #sampled_train_Y = sampled_train_Y.reshape((sampled_train_Y.shape[0],1))




    X_train_reg, Y_train_reg = sklearn.utils.shuffle(sampled_train_X, sampled_train_Y)


    ''''''


    ''''''