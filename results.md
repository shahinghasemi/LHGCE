| desc | emb dim | feature list | folds | optimizer | emb method | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| only BCEWithLogits | 32 | t,e,p,s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 75.5% | 72.5% | 32.9% | 26% | 52.5% | 78.4% | 24%
| nonInteraction 1/k instead of 4/k | 32 | t,e,p,s | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 85% | 72.1% | 13.9% | 14.2% | 36.5% | 86.6% | 9.1% 
|Interactions = nonInteractions| 32 | e,p,s,t | 5 | adam | matrix | 64 | 50 | 0.4 | 0.001 | 70.5% | 72.3% | 70.5% | 70.44% | 86.5% | 41.2% | 59.6%
| all nonInteractions are used in both training and testing while 1/k of interactions are used for testing and k-1/k is used for training| 32 | e,p,t,s | 5 | Adam | matrix | 64 | 50 | 0.4 | 0.001 | 88.7% | 72.2% | 12.2% | 12.2% | 29.7% | 90.2% | 8.1%

[0.12233603 0.72256269 0.12228872 0.88720925 0.2976921  0.90245146 0.08121881]