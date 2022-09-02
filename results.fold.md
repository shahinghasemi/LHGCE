The results below are executed for only **one** fold to examine the best approach available


| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
|dataset='LAGCN', epochs=4000, l=2, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2 | 43.40% | 81.05% | 48.56% | 98.10% | 35.54% | 99.71% | 76.63% |
|dataset='LAGCN', epochs=4000, l=2, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2 + **normalized=TRUE in SAGEConv** | 45.10% | 82.63% | 49.78% | 98.15% | 36.38% | 99.74% | 78.82% |
| aggregator='mean', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr=0.01, n=64, negative_split='all', same=False, thr_percent=2.5 | 37.02% | 82.79% | 43.87% | 97.17% | 43.85% | 98.55% | 43.89% |
|**weightDecay = 0.001**, aggregator='sum', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.01, n=64, negative_split='all', same=False, thr_percent=2.5 |  32.83% | 86.84% | 34.59% | 96.39% | 37.82% | 97.91% | 31.87% |
| aggregator='mean', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.01, n=64, negative_split='all', same=False, thr_percent=2.5 | 25.47% | 84.07% | 27.14% | 96.01% | 29.43% | 97.74% | 25.15% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.01, n=64, negative_split='all', same=False, thr_percent=2.5 | 31.48% | 86.63% | 33.09% | 96.08% | 38.47% | 97.56% | 29.04% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 31.97% | 86.51% | 35.97% | 97.08% | 32.5% | 98.75% | 40.28% |
| weight_decay=0.00001, aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=2000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 27.41% | 86.40% | 32.20% | 96.26% | 35.18% | 97.84% | 29.69% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=2000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 33.40% | 85.98% | 35.98% | 96.77% | 35.94% | 98.34% | 36.01% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=2000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 34.67% | 85.80% | 38.13% | 96.83% | 38.66 % | 98.34% | 37.61% |
[0.34671221 0.85806192 0.38130941 0.96837704 0.38664132 0.98341816
 0.37612256]