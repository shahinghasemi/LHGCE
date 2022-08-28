# variable: n

| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| dataset='LAGCN', epochs=3000, l=2, lr=0.005, n=32, negative_split='all', same=False, thr_percent=3) | 54.27% | 90.4% | 55.34% | 97.97% | 49.54% | 99.22% | 62.83% |
|dataset='LAGCN', epochs=3000, l=2, lr=0.005, n=16, negative_split='all', same=False, thr_percent=3 | 38.26% | 88.81% | 40.85% | 97.25% | 37.39% | 98.8% | 45.26% | 
|**#** dataset='LAGCN', epochs=3000, l=2, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3| 64.94% | 90.52% | 68.29% | 98.41% | 60.96% | 99.37% | 79.11%|
|dataset='LAGCN', epochs=3000, l=2, lr=0.005, n=72, negative_split='all', same=False, thr_percent=3| 43.97% | 87.83% | 45.85% | 96.76% | 46.05% | 98.04% | 47.20% | 
| dataset='LAGCN', epochs=3000, l=2, lr=0.005, n=128, negative_split='all', same=False, thr_percent=3| 45.64% | 88.35% | 48.51% | 97.24% | 47.32% | 98.53% | 51.01% |

# variable: epoch
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| dataset='LAGCN', epochs=1000, l=2, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 55.62% | 90.68% | 56.25% | 97.98% | 51.30% | 99.19% | 62.34% |
| **#** dataset='LAGCN', epochs=2000, l=2, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 69.92% | 91.18% | 71.94% | 98.75% | 63% | 99.67% | 83.93% |
|‌dataset='LAGCN', epochs=2500, l=2, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 61.71% | 89.13% | 64.44% | 98.08% | 57.62% | 99.13% | 74.86% |
| dataset='LAGCN', epochs=3500, l=2, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 62.89% | 90.44% | 66.11% | 98.16% | 61.41% | 99.11% |


# variable: l
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| dataset='LAGCN', epochs=2000, l=1, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 75.76% | 91.7% | 79.83% | 99.11% | 69.16% | 99.89% | 94.39% |
| dataset='LAGCN', epochs=2000, l=3, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 48.67% | 80.96% | 40.09% | 78.14% | 58.58% | 78.64% | 42.31% |



# variable: lr
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| dataset='LAGCN', epochs=2000, l=1, lr=0.5, n=64, negative_split='all', same=False, thr_percent=3 | 50.25% | 50% | 4.91% | 2.52% | 1% | 0% | 2.52% |
| dataset='LAGCN', epochs=2000, l=1, lr=0.05, n=64, negative_split='all', same=False, thr_percent=3 | 43.34% | 89.43% | 43.56% | 96.72% | 46.89% | 98.01% | 43.25% | 
| dataset='LAGCN', epochs=2000, l=1, lr=0.005, n=64, negative_split='all', same=False, thr_percent=3 | 75.76% | 91.7% | 79.83% | 99.11% | 69.16% | 99.89% | 94.39% |
| dataset='LAGCN', epochs=2000, l=1, lr=0.01, n=64, negative_split='all', same=False, thr_percent=3 | 76.81% | 91.46% | 82.73% | 99.24% | 71.43% | 99.96% | 98.3% |

[0.76810516 0.91465693 0.82734675 0.99249151 0.71430899 0.99968409  0.98302426]