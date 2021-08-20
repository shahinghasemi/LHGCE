import torch
from autoencoder import trainAutoEncoders
from prepareData import prepareData
import numpy as np
from FNN import trainFNN

EMBEDDING_DEM = 32
N_DRUGS = 269
N_DISEASES = 598
DRUG_FEATURES = ['structure', 'target', 'enzyme', 'pathway']
N_FEATURES = 269 + 598
N_EPOCHS = 1
N_INTERACTIONS = 18416
N_NON_INTERACTIONS = 142446
N_BATCHSIZE = 1000
FOLDS = 5

def crossValidation(drugSimDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices):
    # To be dividable by 5
    index = np.arange(N_INTERACTIONS - 1)
    np.random.shuffle(index)
    index = index.reshape(FOLDS, N_INTERACTIONS // FOLDS)

    for k in range(FOLDS):
        testInteractionsIndex = index[k]
        trainInteractionsIndex = np.setdiff1d(index.flatten(), testInteractionsIndex, assume_unique=True)

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

            # For encoders we only use interactions data
            trainedAutoEncoder = trainAutoEncoders(XTrain, N_FEATURES, N_EPOCHS, trainInteractionsIndex.shape[0], N_BATCHSIZE)
            
            for drugIndex, diseaseIndex in nonInteractionIndices:
                drug = drugSimDic[drugFeature][drugIndex]
                disease = diseaseSim[diseaseIndex]
                XTrain.append(np.hstack((drug, disease)))
                YTrain.append([0])

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

            for drugIndex, diseaseIndex in nonInteractionIndices:
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

        y_pred = trainedModel(torch.tensor(XTest).float()).detach().numpy()
        y_pred = np.array(y_pred)
        print(y_pred, y_pred.shape)

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
