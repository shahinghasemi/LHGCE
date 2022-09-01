The results below are executed for only **one** fold to examine the best approach available


| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
|dataset='LAGCN', epochs=4000, l=2, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2 | 43.40% | 81.05% | 48.56% | 98.10% | 35.54% | 99.71% | 76.63% |
|dataset='LAGCN', epochs=4000, l=2, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2 + **normalized=TRUE in SAGEConv** | 45.10% | 82.63% | 49.78% | 98.15% | 36.38% | 99.74% | 78.82% |

[0.357102627947822, 0.8181138778931467, 0.421711331791593, 0.9709090660243075, 0.42085257, 0.9851311032328267, 0.4225736]