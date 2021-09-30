| description | emb | feature_list | folds | threshold | batch-auto | batch-model | epoch-auto | epoch-model | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-------------|-----|--------------|-------|-----------|------------|---|----------|------------|-------------|---------|----------|-----|----|--|--|--|--|
| weighted loss function | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.5 | 1000 | 1000 | 10 | 10 | 0.4 | 0.001 | 12% | 51% | - |
| weighted loss function | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 64 | 64 | 1000 | 1000 | 0.4 | 0.001 |11% | 47% | - |
| without weighted loss in the DNN | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 1000 | 1000 | 10 | 10 | 0.4 | 0.001 |88% | 56% | - |
| without weighted loss in the DNN + reduced batch-size | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 128 | 128 | 10 | 10 | 0.4 | 0.001 |86% | 58% | - |
| without weighted loss in the DNN + reduced batch-size | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 64 | 64 | 10 | 10 | 0.4 | 0.001 |88% | 59% | - |
| without weighted loss + DNN batch normalization + DNN weightDecay | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | 0.3 | 128 | 128 | 10 | 10 | 0.4 |0.001 | 11% | 50% | - |