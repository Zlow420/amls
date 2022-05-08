from matplotlib import pyplot as plt
import sklearn.metrics as metrics

def debugging(y_pred, Y_validation):

    validation_type, validation_quality = Y_validation[:, 1], Y_validation[:, 0]


    print("Explained Variance:", metrics.r2_score(validation_quality, y_pred))

    plt.close("all")
    plt.matshow(metrics.confusion_matrix(validation_quality, y_pred))
    plt.show()