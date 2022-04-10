import DataPrep_01 as data
import Model_02 as model

X_train, Y_train, X_test, Y_test, X_validation, Y_validation = data.read_CSV()
train_type, train_quality = Y_train[:,0], Y_train[:,1]
test_type, test_quality = Y_test[:,0], Y_test[:,1]
validation_type, validation_quality = Y_validation[:,0], Y_validation[:,1]
X_train, X_test, X_validation = data.scaler(X_train, X_test, X_validation)
X_train, X_test, X_validation = data.normalize(X_train, X_test, X_validation)

model.logReg(X_train, train_type, X_test, test_type)
model.svclassifier(X_train, train_type, X_test, test_type)

model.MLPreg(X_train, train_quality, X_test, test_quality)