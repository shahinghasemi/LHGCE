first the representation of the drug features are learned using auto-encoders
then we use concatenate(feature, diseaseAssociated) to feed into fully-connected neural network

todo: reduce the learning rate

challenge: when we feed encoder's embedding to DNN, back propagation will affect the encoder's parameters too because the embedding is dependent on the encoder's parameters so in this situation I don't know how to separate the DNN and encoder parameters. 