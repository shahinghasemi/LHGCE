The results below are executed for only **one** fold to examine the best approach available


| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
|dataset='LAGCN', epochs=4000, l=2, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2 | 43.40% | 81.05% | 48.56% | 98.10% | 35.54% | 99.71% | 76.63% |
|dataset='LAGCN', epochs=4000, l=2, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2 + **normalized=TRUE in SAGEConv** | 45.10% | 82.63% | 49.78% | 98.15% | 36.38% | 99.74% | 78.82% |

[0.38296019 0.83600576 0.4539894  0.9725104  0.45343471 0.98593141
 0.45454547]

aggregator='sum', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2.5