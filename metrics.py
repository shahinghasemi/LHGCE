
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import numpy as np

def calculateMetric(real_score, predict_score):
    real_score = real_score.reshape((real_score.shape[0],))
    predict_score = predict_score.reshape((predict_score.shape[0],))
    print('after real_score: ', real_score)
    print('after predict_score: ', predict_score)
    _fpr, _tpr, thresholds = roc_curve(real_score, predict_score)
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    print('max f1: ', f1_score_list[np.argmax(f1_score_list)])
    print('max accuracy: ', accuracy_list[np.argmax(accuracy_list)])
    print('max recall: ', recall_list[np.argmax(recall_list)])
    print('max specificity: ', specificity_list[np.argmax(specificity_list)])
    print('max precision: ', precision_list[np.argmax(precision_list)])

    max_index = np.argmax(f1_score_list)
    print('TP: ', TP[max_index], 'TN: ', TN[max_index], 'FP: ', FP[max_index], 'FN: ', FN[max_index])
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
