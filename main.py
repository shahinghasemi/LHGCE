import torch
from autoencoder import trainAutoEncoders
from prepareData import prepareData
import numpy as np
from FNN import trainFNN
from metrics import calculateMetric
import argparse


EMBEDDING_DEM = 32
N_DRUGS = 269
N_DISEASES = 598
FEATURE_LIST = ['structure', 'target', 'enzyme', 'pathway']
N_INTERACTIONS = 18416
N_NON_INTERACTIONS = 142446
FOLDS = 5
THRESHOLD = 0.50
N_EPOCHS_AUTO=20
N_EPOCHS_MODEL=20
N_BATCHSIZE_MODEL=1000
N_BATCHSIZE_AUTO=1000

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

        allDataTraining = []
        allDataTesting = []
        for drugFeature in FEATURE_LIST:
            XTrain = []
            YTrain = []
            for drugIndex, diseaseIndex in interactionIndices[trainInteractionsIndex]:
                drug = drugSimDic[drugFeature][drugIndex]
                disease = diseaseSim[diseaseIndex]
                XTrain.append(np.hstack((drug, disease)))
                YTrain.append([1])

            for drugIndex, diseaseIndex in nonInteractionIndices[trainNonInteractionsIndex]:
                drug = drugSimDic[drugFeature][drugIndex]
                disease = diseaseSim[diseaseIndex]
                XTrain.append(np.hstack((drug, disease)))
                YTrain.append([0])

            XTrain = np.array(XTrain)
            trainedAutoEncoder = trainAutoEncoders(XTrain, N_EPOCHS_AUTO, N_BATCHSIZE_AUTO)

            XTrain = np.array(XTrain)
            featureEmbeddings = []

            for i in range(len(XTrain)):
                embedding = trainedAutoEncoder.encode(torch.tensor(XTrain[i]).float(), True)
                featureEmbeddings.append(embedding)

            allDataTraining.append(featureEmbeddings)

        XTrain = np.hstack((allDataTraining[0], allDataTraining[1], allDataTraining[2], allDataTraining[3]))
        YTrain = np.array(YTrain)
        dataTrain = np.hstack((XTrain, YTrain))
        trainedModel = trainFNN(dataTrain,  EMBEDDING_DEM * len(FEATURE_LIST), N_EPOCHS_MODEL, N_BATCHSIZE_MODEL)

        # TESTING
        for drugFeature in FEATURE_LIST:
            XTest = []
            YTest = []
            for drugIndex, diseaseIndex in interactionIndices[testInteractionsIndex]:
                drug = drugSimDic[drugFeature][drugIndex]
                disease = diseaseSim[diseaseIndex]
                XTest.append(np.hstack((drug, disease)))
                YTest.append([1])

            for drugIndex, diseaseIndex in nonInteractionIndices[testNonInteractionsIndex]:
                drug = drugSimDic[drugFeature][drugIndex]
                disease = diseaseSim[diseaseIndex]
                XTest.append(np.hstack((drug, disease)))
                YTest.append([0])

            featureEmbeddings = []

            for i in range(len(XTest)):
                embedding = trainedAutoEncoder.encode(torch.tensor(XTest[i]).float(), True)
                featureEmbeddings.append(embedding)

            allDataTesting.append(featureEmbeddings)

        XTest = np.hstack((allDataTesting[0], allDataTesting[1], allDataTesting[2], allDataTesting[3]))
        YTest = np.array(YTest)
        y_pred_prob = trainedModel(torch.tensor(XTest).float()).detach().numpy()
        metrics = calculateMetric(y_pred_prob, YTest, THRESHOLD)

        print('metrics: ', metrics)


def main():
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--emb', help='auto-encoder embedding size',type=int, default=32)
    parser.add_argument('--feature-list', help='the feature list to include',  choices=['structure', 'target', 'enzyme', 'pathway'], default=['structure', 'target', 'enzyme', 'pathway'])
    parser.add_argument('--folds', help='number of folds for cross-validation',type=int,  default=5)
    parser.add_argument('--threshold', help='accuracy threshold',type=float,  default=0.5)
    parser.add_argument('--batch-auto', help='batch-size for auto-encoder',type=int, default=1000)
    parser.add_argument('--batch-model', help='batch-size for DNN',type=int, default=1000)
    parser.add_argument('--epoch-auto', help='number of epochs to train in auto-encoder',type=int, default=20)
    parser.add_argument('--epoch-model', help='number of epochs to train in model',type=int, default=20)

    
    args = parser.parse_args()

    EMBEDDING_DEM = args.emb
    FEATURE_LIST = args.feature_list
    N_EPOCHS_AUTO = args.epoch_auto
    N_EPOCHS_MODEL = args.epoch_model
    N_BATCHSIZE_MODEL = args.batch_model
    N_BATCHSIZE_AUTO = args.batch_auto
    FOLDS = args.folds
    THRESHOLD = args.threshold

    drugSimDic = prepareData()

    # read the interactions matrix
    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
    interactionIndices = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    nonInteractionIndices = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
    results = crossValidation(drugSimDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices)


main()
