| description | emb | feature_list | folds | emb method | batch-auto | batch-model | epoch-auto | epoch-model | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-------------|-----|--------------|-------|-----------|------------|---|----------|------------|-------------|---------|----------|-----|----|--|--|--|--|
| without weighted loss | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | "AE" | 64 | 64 | 10 | 10 | 0.4 | 0.001 | 48.5% | 58% | 23% | 31% | 67% | 46% | 13%
| positive weight loss function with total/nPositives | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | "AE" | 64 | 64 | 10 | 10 | 0.4 | 0.001 | 37.8% | 55.2% | 22.1% | 26.9% | 75% | 33% | 13.3%
| positive weight loss function with nPositives/total | 32 |["structure", "target", "enzyme", "pathway"] | 5 | "AE" | 64 | 64 | 10 | 10 | 0.4 | 0.001 | 54.2% | 59% | 23.2% | 17.8% | 60.5% | 53% | 14.3%

