| description | emb | feature_list | folds | threshold | batch-auto | batch-model | epoch-auto | epoch-model | dropout | accuracy | auc | f1 |
|-------------|-----|--------------|-------|-----------|------------|-------------|------------|-------------|---------|----------|-----|----|
| - | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.5 | 1000 | 1000 | 10 | 10 | 0.4 | 12% | 51% | - |
| - | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 64 | 64 | 1000 | 1000 | 0.4 | 11% | 47% | - |
| without weighted loss in the DNN | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 1000 | 1000 | 10 | 10 | 0.4 | 88% | 56% | - |
| without weighted loss in the DNN + reduced batch-size | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 128 | 128 | 10 | 10 | 0.4 | 86% | 58% | - |
