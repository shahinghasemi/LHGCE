| desc | feature list | folds | optimizer | emb method | aggregation | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|--|-|-|-|-|
| DNN OCC using 80% of positives and 50% of negatives | e, p, t, s | 5 | adam | matrix | concatenate | 64 | 50 | 0.4 | 0.001 | 21.4 % | 48.5% | 34.1% | 45.9% | 99.1% | 13.2% | 20.6%
| DNN OCC using 80% of positives and 50% of negatives | e, p, t, s | 5 | adam | jaccard| concatenate | 64 | 50 | 0.4 | 0.001 | 21.4% | 48.5% | 34.1% | 45.9% | 99.1% | 13.2% | 20.6%
| Reconstruction OCC using 80% of positives | e, p, t, s | 5 | adam | matrix |concatenate | 64 | 50 | 0.4 | 0.001 | 91.7% | 65.9% | 10.3% | 5.6% | 19% | 93.6% | 7.1%
| Reconstruction OCC using 80% of negatives | e, p, t, s | 5 | adam | matrix |concatenate| 64 | 50 | 0.4 | 0.001 | 39.7% | 52.1% | 56.3% | 40.2% | 99.2% | 1.3% | 39.3% |
| SVM OCC using 80% of positives | e, p, t, s | 5 | - | matrix | concatenate |  - | - | - | - | 6.1% | - | 5% | - | 98.8% | 3.7% | 2.5% 
| SVM OCC using 80% of positives | t | 5 | - | matrix | concatenate |  - | - | - | - | 10.1% | - | 5.2% | - | 98.1% | 7.8% | 2.6% 
| SVM OCC using 80% of positives | e | 5 | - | matrix | concatenate |  - | - | - | - | 6.9% | - | 5% | - | 98.6% | 4.5% | 2.6% 
| SVM OCC using 80% of positives | p | 5 | - | matrix | concatenate |  - | - | - | - | 9% | - | 5.1% | - | 98.3% | 6.7% | 2.6% 
| SVM OCC using 80% of positives | s | 5 | - | matrix | concatenate |  - | - | - | - | 6% | - | 5% | - | 98.8% | 3.6% | 2.5% 
| SVM OCC using 80% of positives | e, p, t, s | 5 | - | jaccard | concatenate |  - | - | - | - | 6.1% | - | 5% | - | 98.8% | 3.7% | 2.5% 
| SVM OCC using 80% of positives | t | 5 | - | jaccard | concatenate |  - | - | - | - | 10.1% | - | 5.2% | - | 98.1% | 7.8% | 2.6% 

[0.05204746 0.0985588  0.98186261 0.07572045 0.02673226 0. 0.        ]