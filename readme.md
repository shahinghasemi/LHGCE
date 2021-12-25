first the representation of the drug features are learned using auto-encoders
then we use concatenate(feature, diseaseAssociated) to feed into fully-connected neural network


challenge: when we feed encoder's embedding to DNN, back propagation will affect the encoder's parameters too because the embedding is dependent on the encoder's parameters so in this situation I don't know how to separate the DNN and encoder parameters. 

todo: treat involved diseases like drug features in terms of an independent feature that need its own encoder...

challenges: 
- how to implement link prediction using GCN
- how to create the adjacency matrix which play it's role as expected (according to GCN paradigm)

[aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]

EPOCHS: 1500
[0.4323990814811637, 0.9032814914307487, 0.4543014550077695, 0.9735642724187014, 0.4366006, 0.9874477868651058, 0.47349823]

EPOCHS: 2000
[0.4566820908995123, 0.9053106697209743, 0.4694229112833764, 0.9747071060987628, 0.44393158, 0.9884306223454666, 0.4980201]
