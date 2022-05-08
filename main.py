import DataPrep_01 as dataPrep
import Model_02 as model
import Tuning_03 as tuning
import Debugging_04 as debug
import Parallel_05 as parallel

import numpy as np
import time
from matplotlib import pyplot as plt
import argparse


if __name__ ==  '__main__':

    parser = argparse.ArgumentParser(description='Parallelization.')

    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    if args.parallel:
        parallelization = True
    else:
        parallelization = False

    print(parallelization)


    

    # DATAPREP
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = dataPrep.read_CSV()

    # MODEL
    #y_pred_class_logreg, y_pred_class_svm, y_pred_reg_mlp, y_pred_reg_knn = model.basic_models(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)

    # TUNING
    if(not parallelization):
        t = time.time()
        knnResult, mlpResult, knnModel, mlpModel = tuning.perform_tuning(X_train, Y_train, X_validation, Y_validation)
        elapsed = time.time() - t
        print("Elapsed Time:", elapsed)


    # PARALELLIZATION
    if(parallelization):
        t = time.time()
        knnResult, mlpResult, knnModel, mlpModel = parallel.perform_tuning(X_train, Y_train, X_validation, Y_validation, 2)
        elapsed = time.time() - t
        print("Elapsed Time:", elapsed)

    # DEBUGGING

    debug.debugging(mlpResult, Y_validation)








