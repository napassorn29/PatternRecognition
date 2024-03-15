import numpy as np

def classification_report(y_actual: np.ndarray, y_predict: np.ndarray,flag:str = None):
# Description: Calculate accuracy, precision, recall, and F1-score based on given [y_actual], and [y_predict].
#              Display confusion matrix and results.
    
    # -- |Local variables initialization| --
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    # -- |Parameter counts| --
    for y_a, y_p in zip(y_actual, y_predict):
        if y_p[0] == 1 and y_a == y_p:
            TP = TP + 1
        elif y_p[0] == 1 and y_a != y_p:
            FP = FP + 1
        elif y_p[0] == 0 and y_a == y_p:
            TN = TN + 1
        elif y_p[0] == 0 and y_a != y_p:
            FN = FN + 1
    
    # -- |Results calculation| --
    accuracy = (TP + TN)/(TP + FP + TN + FN)

    if TP + FP == 0:
        precision = np.NaN
    else:
        precision = TP/(TP + FP)
    
    if TP + FN == 0:
        recall = np.NaN
    else:
        recall = TP/(TP + FN)
        
    f1 = 2*(precision*recall)/(precision + recall)

    if flag == "Matrix":
        # -- |Results display| --
        print('Confusion Matrix')
        print('----------------------------------')
        print('|            |     Predicted     |')
        print('|            ---------------------   Accuracy  = {0:}'.format(accuracy))
        print('|            |    P    |    N    |')
        print('----------------------------------   Precision = {0:}'.format(precision))
        print("|        | P |   {0:^3d}   |   {1:^3d}   |   Recall    = {2:}".format(TP, FN, recall))
        print('| Actual -------------------------')
        print("|        | N |   {0:^3d}   |   {1:^3d}   |   F1-score  = {2:}".format(FP, TN, f1))
        print('----------------------------------')

    return [accuracy, precision, recall, f1]

def MSE(y_actual: np.ndarray, y_predict: np.ndarray):
# Description: Calculate the mean square error of [y_actual], and [y_predict].
    
    # -- |MSE calculation| --
    MSE = np.sum((y_actual - y_predict)**2)/y_actual.shape[0]

    return MSE