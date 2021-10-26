| desc | feature list | folds | optimizer | emb method | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| DNN OCC using 80% of positives and 50% of negatives | e, p, t, s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 21.4 % | 48.5% | 34.1% | 45.9% | 99.1% | 13.2% | 20.6%
| DNN OCC using 80% of positives and 50% of negatives | e, p, t, s | 5 | adam | jaccard | 64 | 50 | 0.4 | 0.001 | 21.4% | 48.5% | 34.1% | 45.9% | 99.1% | 13.2% | 20.6%
| Reconstruction OCC using 80% of positives | e, p, t, s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 91.7% | 65.9% | 10.3% | 5.6% | 19% | 93.6% | 7.1%
| Reconstruction OCC using 80% of negatives | e, p, t, s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 39.7% | 52.1% | 56.3% | 40.2% | 99.2% | 1.3% | 39.3% |


[0.40271522 0.52148525 0.56393297 0.3975823  0.99214771 0.01326126 0.39393675]