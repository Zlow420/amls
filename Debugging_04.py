from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd

def debugging(y_pred_mlp, y_pred_knn, Y_validation):

    validation_type, validation_quality = Y_validation[:, 1], Y_validation[:, 0]

    data_df = pd.DataFrame({'validation': validation_quality, 'mlp': y_pred_mlp, 'knn': y_pred_knn})
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data_df, kind='violin')
    plt.title("Violinplot")
    g.set_axis_labels("", "quality")
    plt.savefig("figures/[VIOLINPLOT QUALITY]")
    plt.close()

    r2_mlp = [metrics.r2_score(validation_quality, y_pred_mlp), 'r2', 'mlp']
    r2_knn = [metrics.r2_score(validation_quality, y_pred_knn), 'r2', 'knn']
    mse_mlp = [metrics.mean_squared_error(validation_quality, y_pred_mlp), 'mse', 'mlp']
    mse_knn = [metrics.mean_squared_error(validation_quality, y_pred_knn), 'mse', 'knn']
    rmse_mlp = [metrics.mean_squared_error(validation_quality, y_pred_mlp, squared=False), 'rmse', 'mlp']
    rmse_knn = [metrics.mean_squared_error(validation_quality, y_pred_knn, squared=False), 'rmse', 'knn']
    mae_mlp = [metrics.mean_absolute_error(validation_quality, y_pred_mlp), 'mae', 'mlp']
    mae_knn = [metrics.mean_absolute_error(validation_quality, y_pred_knn), 'mae', 'knn']

    mlp_res = [r2_mlp[0], mse_mlp[0], rmse_mlp[0], mae_mlp[0]]
    knn_res = [r2_knn[0], mse_knn[0], rmse_knn[0], mae_knn[0]]
    #print("The best performing metric for mlp is: " + str(mlp_res[mlp_res.index(max(mlp_res))][1]) + " with the score " + str(max(mlp_res)))
    #print("The best performing metric for knn is: " + str(knn_res[knn_res.index(max(knn_res))][1]) + " with the score " + str(max(knn_res)))

    metrics_df = pd.DataFrame([r2_mlp, r2_knn, mse_mlp, mse_knn, rmse_mlp, rmse_knn, mae_mlp, mae_knn], columns=['score', 'type', 'model'])

    plt.figure(figsize=(10, 10))
    g = sns.catplot(
        data=metrics_df, kind="bar",
        x="type", y="score", hue="model",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "score")
    g.legend.set_title("Models")
    plt.title("Barplot")
    plt.savefig("figures/[BARPLOT METRICS]")
    plt.close()

    #fig, ax = plt.subplots(1, 1)
    plt.matshow(metrics.confusion_matrix(validation_quality, y_pred_mlp))
    #ax.set_xticks([i for i in range(int(min(validation_quality)),int(max(validation_quality))+1)])
    #ax.set_yticks([i for i in range(int(min(validation_quality)),int(max(validation_quality))+1)])
    plt.title("Correlation Matrix")
    plt.xlabel('validation')
    plt.ylabel('mlp prediction')
    plt.savefig("figures/[CORRELATION MATRIX VALIDATION MLP]")
    plt.close()

    #fig, ax = plt.subplots(1, 1)
    plt.matshow(metrics.confusion_matrix(validation_quality, y_pred_knn))
    #ax.set_xticks([i for i in range(int(min(validation_quality)), int(max(validation_quality))+1)])
    #ax.set_yticks([i for i in range(int(min(validation_quality)), int(max(validation_quality))+1)])
    plt.title("Correlation Matrix")
    plt.xlabel('validation')
    plt.ylabel('knn prediction')
    plt.savefig("figures/[CORRELATION MATRIX VALIDATION KNN]")
    plt.close()

