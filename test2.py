from sklearn.metrics import accuracy_score, roc_curve, auc

acc = accuracy_score([[1], [1], [0]], [[1], [0], [0]])
print(acc)