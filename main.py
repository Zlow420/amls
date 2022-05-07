import DataPrep_01 as dataPrep
import Model_02 as model
import Tuning_03 as tuning

# DATAPREP
X_train, Y_train, X_test, Y_test, X_validation, Y_validation = dataPrep.read_CSV()

# MODEL
#y_pred_class_logreg, y_pred_class_svm, y_pred_reg_mlp, y_pred_reg_knn = model.basic_models(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)

# TUNING

tuning.try_nonlins(X_train, Y_train, X_validation, Y_validation)



