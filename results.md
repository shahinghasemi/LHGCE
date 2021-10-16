| desc | emb dim | feature list | folds | optimizer | emb method | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| only BCEWithLogits | 32 | t,e,p,s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 75.5% | 72.5% | 32.9% | 26% | 52.5% | 78.4% | 24%
| nonInteraction 1/k instead of 4/k | 32 | t,e,p,s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 85% | 72.1% | 13.9% | 14.2% | 36.5% | 86.6% | 9.1% 

[0.1425323  0.72103048 0.13902422 0.85036595 0.36508281 0.86605005 0.09162082]