
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np

def calculateMetric(y_prob, y_true, threshold):
    y_pred = []
    for prob in y_prob:
        if prob >= threshold:
            y_pred.append([1])
        else:
            y_pred.append([0])

    y_pred_label = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred_label)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    aucScore = auc(fpr, tpr)
    return {
        'acc': acc,
        'auc': aucScore,
    }
