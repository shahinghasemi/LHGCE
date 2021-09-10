import torch
from autoencoder import trainAutoEncoders
from prepareData import prepareData
import numpy as np
from FNN import trainFNN
from metrics import calculateMetric

EMBEDDING_DEM = 32
N_DRUGS = 269
N_DISEASES = 598
DRUG_FEATURES = ['structure', 'target', 'enzyme', 'pathway']
N_FEATURES = 269 + 598
N_EPOCHS = 20
N_INTERACTIONS = 18416
N_NON_INTERACTIONS = 142446
N_BATCHSIZE = 1000
FOLDS = 5
THRESHOLD = 0.50

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
        print('trainInteractionsIndex: ', trainInteractionsIndex.shape)

        testNonInteractionsIndex = totalNonInteractionIndex[k]
        trainNonInteractionsIndex = np.setdiff1d(totalNonInteractionIndex.flatten(), testNonInteractionsIndex, assume_unique=True)
        print('trainNonInteractionsIndex: ', trainNonInteractionsIndex.shape)

        allDataTraining = []
        allDataTesting = []
        for drugFeature in DRUG_FEATURES:
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

            trainedAutoEncoder = trainAutoEncoders(XTrain, N_FEATURES, N_EPOCHS, N_BATCHSIZE)

            XTrain = np.array(XTrain)
            featureEmbeddings = []

            for i in range(len(XTrain)):
                embedding = trainedAutoEncoder.encode(torch.tensor(XTrain[i]).float(), True)
                featureEmbeddings.append(embedding)

            allDataTraining.append(featureEmbeddings)

        XTrain = np.hstack((allDataTraining[0], allDataTraining[1], allDataTraining[2], allDataTraining[3]))
        YTrain = np.array(YTrain)
        dataTrain = np.hstack((XTrain, YTrain))
        trainedModel = trainFNN(dataTrain,  EMBEDDING_DEM * 4, N_EPOCHS, N_BATCHSIZE)

        # TESTING
        for drugFeature in DRUG_FEATURES:
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
    drugSimDic = prepareData()

    # read the interactions matrix
    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
    interactionIndices = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    nonInteractionIndices = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
    # training the autoencoders
    # models = trainAutoEncoders(mixedData,N_DRUG_FEATURES, N_FEATURES, N_EPOCHS, N_INTERACTIONS, N_BATCHSIZE)
    results = crossValidation(drugSimDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices)


main()
