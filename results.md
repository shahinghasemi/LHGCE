| desc | messageEdge | supervisionEdge | testEdge | feature list | folds | optimizer | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|undirected graph| allEdges | interactions that != interactions[k] | interactions[k]| [t, p, e, s] | 5 | adam | batch gradient |1500 | 0 | 0.01 | 97.3% | 90.3% | 45.4% | 43.2% | 43.6% | 98.7% | 47.3% |
|undirected graph| allEdges | interactions that != interactions[k] | interactions[k]| [t, p, e, s] | 5 | adam | batch gradient |2000 | 0 | 0.01 | 97.4% | 90.5% | 46.9% | 45.6% | 44.3% | 98.8% | 49.8% |
|undirected graph|  interactions that != testEdges U supervisionEdges | disjoint to messageEdges (interactions[k+1]) | interactions[k]| [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96% | 80.6% | 25% | 20.6% | 26.2% | 97.8% | 24.5%
|undirected graph | interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96.7% | 87.6% | 35.4% | 32.1% | 34.8% | 98.3% | 36.5% 
|undirected graph| allEdges | interactions that != interactions[k] | interactions[k]| [t, p, e, s] | 5 | adam | batch gradient |1500 | 0 | 0.01 | 96.3% | 87.7% | 33.4% | 30.4% | 36.5% | 97.8% | 31.4% 
