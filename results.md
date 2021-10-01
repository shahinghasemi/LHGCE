| description | emb | feature_list | folds | emb method | batch-auto | batch-model | epoch-auto | epoch-model | dropout | LR | accuracy | auc | f1 | aupr | recall | specificity | precision | 
|-------------|-----|--------------|-------|-----------|------------|---|----------|------------|-------------|---------|----------|-----|----|--|--|--|--|
| without weighted loss | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | "AE" | 64 | 64 | 10 | 10 | 0.4 | 0.001 | 48.5% | 58% | 23% | 31% | 67% | 46% | 13%
| positive weight loss function with total/nPositives | 32 | ["structure", "target", "enzyme", "pathway"] | 5 | "AE" | 64 | 64 | 10 | 10 | 0.4 | 0.001 | 37.8% | 55.2% | 22.1% | 26.9% | 75% | 33% | 13.3%
| positive weight loss function with nPositives/total | 32 |["structure", "target", "enzyme", "pathway"] | 5 | "AE" | 64 | 64 | 10 | 10 | 0.4 | 0.001 | 54.2% | 59% | 23.2% | 17.8% | 60.5% | 53% | 14.3%
| positive weight loss function with nPositives/tota | - | ["structure", "target", "enzyme", "pathway"] | 5 | "jaccard" | - | 64 | - | 10 | 0.4 | 0.001 | 77.1% | 71.4% | 32% | 29.1% | 47% | 81% | 24.5%


[0.39900089 0.62757223 0.27198344 0.51166853 0.6702688  0.49116501 0.19534836]
[0.29011294 0.71292713 0.31983241 0.78182892 0.44805865 0.82497806 0.25080895]

with epoch 50   
[0.36523598 0.74520254 0.3539599  0.7962514  0.48645126 0.83630173 0.2794964 ]
[0.42422782 0.6948068  0.32243799 0.64074972 0.62861797 0.64231809 0.23576169]

with 500 epoch