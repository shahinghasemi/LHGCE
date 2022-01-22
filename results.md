| desc | messageEdge | supervisionEdge | testEdge | feature list | folds | optimizer | batch size | epoch | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|undirected graph|  interactions that != testEdges U supervisionEdges | disjoint to messageEdges (interactions[k+1]) | interactions[k]| [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96% | 80.6% | 25% | 20.6% | 26.2% | 97.8% | 24.5%
|undirected graph | interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96.7% | 87.6% | 35.4% | 32.1% | 34.8% | 98.3% | 36.5% 
|undirected graph + weighted BCE | interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 96.2% | 89.8% | 35.3% | 31.4% | 40.8% | 96.6% | 31.5
|(only reverse(drug,disease) edge exist + weighted BCE | interactions that != interactions[k] | interactions != interactions[k] (the same as message edges) | interactions[k] | [t, p, e, s] | 5 | adam | batch gradient | 500 | 0 | 0.01 | 95.8% | 89.7% | 33.4% | 29.8% | 41.4% | 97.2% | 28.1% 
| undirected graph | interactions that != interactions[k] | interactions that != interactions[k] (the same as message edges) | interactions[k] | [e, p, t, s] | 5 | adam | batch gradient | 2000 | 0 | 0.01 | 97.8% | 89.8% | 53.4% | 52.1% | 48.7% | 99.1% | 59.2% |
| undirected graph | interactions that != interactions[k] | interactions that != interactions[k] (the same as message edges) | interactions[k] | [e, p, t, s] | 5 | adam | batch gradient | 2500 | 0 | 0.01 | 98% | 89.8% | 55.4% | 54.2% | 49.3% | 99.2% | 63.5% |
| undirected graph | interactions that != interactions[k] | interactions that != interactions[k] (the same as message edges) | interactions[k] | [e, p, t, s] | 5 | adam | batch gradient | 3000 | 0 | 0.01 | 98% | 90% | 56.8% | 55.6% | 51.4% | 99.2% | 63.5% 
| undirected graph, non-interactions are splited in 1/k and k-1/k | interactions that != interactions[k] | interactions that != interactions[k] (the same as message edges) | interactions[k] | [e, p, t, s] | 5 | adam | batch gradient | 3000 | 0 | 0.01 



