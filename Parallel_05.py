import Tuning_03 as tuning
from joblib import parallel_backend

def perform_tuning(X_train, Y_train, X_validation, Y_validation, numberOfJobs):

    with parallel_backend('threading', n_jobs=numberOfJobs):
        return tuning.perform_tuning(X_train, Y_train, X_validation, Y_validation, parallel=True)
