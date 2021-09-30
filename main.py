import torch
from autoencoder import trainAutoEncoders
from prepareData import prepareData
import numpy as np
from FNN import trainFNN
from metrics import calculateMetric
import argparse

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--emb', help='auto-encoder embedding size',type=int, default=32)
parser.add_argument('--feature-list', help='the feature list to include', type=str, nargs="+")
parser.add_argument('--folds', help='number of folds for cross-validation',type=int,  default=5)
parser.add_argument('--batch-auto', help='batch-size for auto-encoder',type=int, default=1000)
parser.add_argument('--batch-model', help='batch-size for DNN',type=int, default=1000)
parser.add_argument('--epoch-auto', help='number of epochs to train in auto-encoder',type=int, default=20)
parser.add_argument('--epoch-model', help='number of epochs to train in model',type=int, default=20)
parser.add_argument('--dropout', help='dropout probability for DNN',type=float, default=0.3)
parser.add_argument('--lr-model', help='learning rate for DNN',type=float, default=0.001)
parser.add_argument('--weight-decay-model', help='weight decay value for DNN optimizer',type=float, default=0.3)
args = parser.parse_args()
print(args)
EMBEDDING_DEM = args.emb
FEATURE_LIST = args.feature_list
N_INTERACTIONS = 18416
N_NON_INTERACTIONS = 142446
N_EPOCHS_AUTO = args.epoch_auto
N_EPOCHS_MODEL = args.epoch_model
N_BATCHSIZE_MODEL = args.batch_model
N_BATCHSIZE_AUTO = args.batch_auto
FOLDS = args.folds
DROPOUT = args.dropout
LEARNING_RATE_MODEL = args.lr_model
WEIGHT_DECAY_MODEL = args.weight_decay_model

def crossValidation(drugSimDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices):
    # To be dividable by 5
    totalInteractionIndex = np.arange(N_INTERACTIONS - 1)
    totalNonInteractionIndex = np.arange(N_NON_INTERACTIONS - 1)

    np.random.shuffle(totalInteractionIndex)
    np.random.shuffle(totalNonInteractionIndex)

    totalInteractionIndex = totalInteractionIndex.reshape(FOLDS, N_INTERACTIONS // FOLDS)
    totalNonInteractionIndex = totalNonInteractionIndex.reshape(FOLDS, N_NON_INTERACTIONS // FOLDS)

    for k in range(FOLDS):
        testInteractionsIndex = totalInteractionIndex[k]
        trainInteractionsIndex = np.setdiff1d(totalInteractionIndex.flatten(), testInteractionsIndex, assume_unique=True)

        testNonInteractionsIndex = totalNonInteractionIndex[k]
        trainNonInteractionsIndex = np.setdiff1d(totalNonInteractionIndex.flatten(), testNonInteractionsIndex, assume_unique=True)

        autoEncoders = []
        allFeatureEmbeddings_train = []
        allFeatureEmbeddings_test = []
        for featureIndex in range(len(FEATURE_LIST)):
            XTrain = []
            YTrain = []
            for drugIndex, diseaseIndex in interactionIndices[trainInteractionsIndex]:
                drug = drugSimDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                YTrain.append([1])

            for drugIndex, diseaseIndex in nonInteractionIndices[trainNonInteractionsIndex]:
                drug = drugSimDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                YTrain.append([0])

            XTrain = np.array(XTrain)
            autoEncoders.append(trainAutoEncoders(XTrain, N_EPOCHS_AUTO, N_BATCHSIZE_AUTO))

            XTrain = np.array(XTrain)
            featureEmbeddings = []
            for i in range(len(XTrain)):
                embedding = autoEncoders[featureIndex].encode(torch.tensor(XTrain[i]).float(), True)
                featureEmbeddings.append(embedding)

            allFeatureEmbeddings_train.append(featureEmbeddings)


        involvedDiseases = []
        for drugIndex, diseaseIndex in interactionIndices[trainInteractionsIndex]:
            involvedDiseases.append(diseaseSim[diseaseIndex])

        for drugIndex, diseaseIndex in nonInteractionIndices[trainNonInteractionsIndex]:
            involvedDiseases.append(diseaseSim[diseaseIndex])
        
        XTrain = []
        for i in range(len(FEATURE_LIST)):
            if i == 0:
                XTrain = allFeatureEmbeddings_train[i]
            else:
                XTrain = np.hstack((XTrain, allFeatureEmbeddings_train[i]))
    
        XTrain = np.hstack((XTrain, involvedDiseases))
        YTrain = np.array(YTrain)
        dataTrain = np.hstack((XTrain, YTrain))
        trainedModel = trainFNN(dataTrain, N_EPOCHS_MODEL, N_BATCHSIZE_MODEL, DROPOUT, LEARNING_RATE_MODEL)

        # TESTING
        for featureIndex in range(len(FEATURE_LIST)):
            XTest = []
            YTest = []
            for drugIndex, diseaseIndex in interactionIndices[testInteractionsIndex]:
                drug = drugSimDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                YTest.append([1])

            for drugIndex, diseaseIndex in nonInteractionIndices[testNonInteractionsIndex]:
                drug = drugSimDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                YTest.append([0])

            featureEmbeddings = []
            for i in range(len(XTest)):
                embedding = autoEncoders[featureIndex].encode(torch.tensor(XTest[i]).float(), True)
                featureEmbeddings.append(embedding)

            allFeatureEmbeddings_test.append(featureEmbeddings)

        involvedDiseases = []
        for drugIndex, diseaseIndex in interactionIndices[testInteractionsIndex]:
            involvedDiseases.append(diseaseSim[diseaseIndex])

        for drugIndex, diseaseIndex in nonInteractionIndices[testNonInteractionsIndex]:
            involvedDiseases.append(diseaseSim[diseaseIndex])

        XTest = []
        for i in range(len(FEATURE_LIST)):
            if i == 0:
                XTest = allFeatureEmbeddings_test[i]
            else:
                XTest = np.hstack((XTest, allFeatureEmbeddings_test[i]))

        XTest = np.hstack((XTest, involvedDiseases))
        YTest = np.array(YTest)
        y_pred_prob = trainedModel(torch.tensor(XTest).float()).detach().numpy()
        metrics = calculateMetric(YTest, y_pred_prob)

        print('metrics: ', metrics)

def main():
    drugSimDic = prepareData()

    # read the interactions matrix
    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
    interactionIndices = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    nonInteractionIndices = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
    results = crossValidation(drugSimDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices)

main()
