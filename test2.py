from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
acc = accuracy_score([0, 0, 1, 1, 0], [0.3, 0.4, 0.1, 0.9, 0.25])
print('acc: ', acc )