import DataPrep_01 as dataPrep
import Model_02 as model
import Tuning_03 as tuning

# DATAPREP
X_train, Y_train, X_test, Y_test, X_validation, Y_validation = dataPrep.read_CSV()

X_train, Y_train, X_test, Y_test, X_validation, Y_validation = model.removeOutliers(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)

train_type, train_quality = Y_train[:,1], Y_train[:,0]
test_type, test_quality = Y_test[:,1], Y_test[:,0]
validation_type, validation_quality = Y_validation[:,1], Y_validation[:,0]

# MODEL
# classification task
model.logReg(X_train, train_type, X_test, test_type)
model.svclassifier(X_train, train_type, X_test, test_type)

# standardization
X_train, X_test, X_validation = model.scaler(X_train, X_test, X_validation)
model.logReg(X_train, train_type, X_test, test_type)
model.svclassifier(X_train, train_type, X_test, test_type)

# normalization
X_train, X_test, X_validation = model.normalize(X_train, X_test, X_validation)
model.logReg(X_train, train_type, X_test, test_type)
model.svclassifier(X_train, train_type, X_test, test_type)


# regression task
#model.MLPreg(X_train, train_quality, X_test, test_quality)

model.MLPregression(X_train, train_quality.reshape((train_quality.shape[0], 1)), X_test, test_quality)



