| desc | messageEdge | supervisionEdge | testEdge | feature list | folds | optimizer | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|undirected graph|  interactions that != testEdges U supervisionEdges | disjoint to messageEdges (interactions[k+1]) | interactions[k]| [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96% | 80.6% | 25% | 20.6% | 26.2% | 97.8% | 24.5%
|undirected graph | interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96.7% | 87.6% | 35.4% | 32.1% | 34.8% | 98.3% | 36.5% 
|undirected graph + weighted BCE | interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96.2% | 89.8% | 35.3% | 31.4% | 40.8% | 96.6% | 31.5
|undirected graph + weighted BCE + (only reverse(drug,disease) edge exist)| interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 95.8% | 89.7% | 33.4% | 29.8% | 41.4% | 97.2% | 28.1% 

