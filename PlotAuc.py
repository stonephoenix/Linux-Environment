from sklearn.metrics import roc_curve, auc
from ConfusionMatrix import ConfusionMatrix
from matplotlib import pyplot as plt


# y_actual is 1 d array, y_score is 1d array, should be True probabilty
def plot_auc(y_actual_train, y_score_train,
             y_actual_val=None, y_score_val=None,
             y_actual_test=None, y_score_test=None,
             do_print=True):
    dfs = []

    # Compute ROC curve and ROC area for each class
    fpr, tpr, splitvalue = roc_curve(y_actual_train, y_score_train)
    roc_auc = auc(fpr, tpr)
    # df1 = pd.DataFrame({'splitvalue': splitvalue, 'fpr': fpr, 'tpr': tpr})
    df1 = ConfusionMatrix(y_score_train, y_actual_train)
    dfs.append(df1)

    roc_auc2 = None
    if ((y_actual_val is not None) and (y_score_val is not None)):
        fpr2, tpr2, _ = roc_curve(y_actual_val, y_score_val)
        roc_auc2 = auc(fpr2, tpr2)
        df2 = ConfusionMatrix(y_score_val, y_actual_val)
        dfs.append(df2)

    roc_auc3 = None
    if ((y_actual_test is not None) and (y_score_test is not None)):
        fpr3, tpr3, _ = roc_curve(y_actual_test, y_score_test)
        roc_auc3 = auc(fpr3, tpr3)
        df3 = ConfusionMatrix(y_score_test, y_actual_test)
        dfs.append(df3)

    if(not do_print):
        return roc_auc, roc_auc2, roc_auc3, dfs

    # print
    if ((y_actual_test is not None) and (y_score_test is not None)):
        fig = plt.figure(figsize=(18, 5))
        n = 3
    elif ((y_actual_val is not None) and (y_score_val is not None)):
        fig = plt.figure(figsize=(12, 5))
        n = 2
    else:
        fig = plt.figure(figsize=(12, 5))
        n = 1

    ax1 = fig.add_subplot(1, n, 1)
    _ax_setfigure(ax1, fpr, tpr, roc_auc, "Train Data")
    if ((y_actual_val is not None) and (y_score_val is not None)):
        ax2 = fig.add_subplot(1, n, 2)
        _ax_setfigure(ax2, fpr2, tpr2, roc_auc2, "Validate Data")
    if ((y_actual_val is not None) and (y_score_val is not None)):
        ax3 = fig.add_subplot(1, n, 3)
        _ax_setfigure(ax3, fpr3, tpr3, roc_auc3, "Test/Holdout Data")
    return roc_auc, roc_auc2, roc_auc3, dfs


def _ax_setfigure(axi, fpr, tpr, roc_auc, notes=""):
    lw = 2
    axi.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    axi.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axi.set_xlim([0.0, 1.0])
    axi.set_ylim([0.0, 1.05])
    axi.set_xlabel('False Positive Rate')
    axi.set_ylabel('True Positive Rate')
    axi.set_title('ROC:' + notes)
    axi.legend(loc="lower right")
