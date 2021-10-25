| desc | feature list | folds | optimizer | emb method | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| default | e, p, t, s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 21.4 % | 48.5% | 34.1% | 45.9% | 99.1% | 13.2% | 20.6%
| default | e, p, t, s | 5 | adam | jaccard | 64 | 50 | 0.4 | 0.001 | 21.4% | 48.5% | 34.1% | 45.9% | 99.1% | 13.2% | 20.6%
| Reconstruction OCC using positives | e, p, t, s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 91.7% | 65.9% | 10.3% | 5.6% | 19% | 93.6% | 7.1%

[0.05644127 0.65909421 0.10376771 0.91743403 0.19049688 0.93622942 0.07152761]