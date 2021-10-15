import torch
from prepareData import prepareData
import numpy as np
from FNN import trainFNN, testFNN
from metrics import calculateMetric
import argparse

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

# Command line options
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--emb', help='auto-encoder embedding size',type=int, default=32)
parser.add_argument('--emb-method', help='embedding method for drug features', type=str, default='matrix')
parser.add_argument('--feature-list', help='the feature list to include', type=str, nargs="+")
parser.add_argument('--folds', help='number of folds for cross-validation',type=int,  default=5)
parser.add_argument('--batchsize', help='batch-size for DNN',type=int, default=1000)
parser.add_argument('--epoch', help='number of epochs to train in model',type=int, default=20)
parser.add_argument('--dropout', help='dropout probability for DNN',type=float, default=0.3)
parser.add_argument('--lr', help='learning rate for DNN',type=float, default=0.001)
parser.add_argument('--agg-method', help='aggregation method for DNN input', type=str, default='concatenate')

args = parser.parse_args()
print(args)

# Setting the global variables
EMBEDDING_DEM = args.emb
FEATURE_LIST = args.feature_list
EPOCHS = args.epoch
BATCHSIZE = args.batchsize
EMBEDDING_METHOD = args.emb_method
FOLDS = args.folds
DROPOUT = args.dropout
LEARNING_RATE= args.lr
AGGREGATE_METHOD = args.agg_method
DRUG_NUMBER = 269
DISEASE_NUMBER = 598
ENZYME_NUMBER = 108
STRUCTURE_NUMBER = 881
PATHWAY_NUMBER = 258
TARGET_NUMBER = 529
INTERACTIONS_NUMBER = 18416
NONINTERACTIONS_NUMBER = 142446

def crossValidation(drugDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices):
    metrics = np.zeros(7)
    # To be dividable by 5
    totalInteractionIndex = np.arange(INTERACTIONS_NUMBER - 1)
    totalNonInteractionIndex = np.arange(NONINTERACTIONS_NUMBER - 1)

    np.random.shuffle(totalInteractionIndex)
    np.random.shuffle(totalNonInteractionIndex)

    totalInteractionIndex = totalInteractionIndex.reshape(FOLDS, INTERACTIONS_NUMBER // FOLDS)
    totalNonInteractionIndex = totalNonInteractionIndex.reshape(FOLDS, NONINTERACTIONS_NUMBER // FOLDS)

    for k in range(FOLDS):
        testInteractionsIndex = totalInteractionIndex[k]
        trainInteractionsIndex = np.setdiff1d(totalInteractionIndex.flatten(), testInteractionsIndex, assume_unique=True)

        testNonInteractionsIndex = totalNonInteractionIndex[k]
        trainNonInteractionsIndex = np.setdiff1d(totalNonInteractionIndex.flatten(), testNonInteractionsIndex, assume_unique=True)

        allDataDic = {}
        for featureIndex in range(len(FEATURE_LIST)):
            involvedDiseases = []
            XTrain = []
            YTrain = []
            for drugIndex, diseaseIndex in interactionIndices[trainInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTrain.append([1])
            
            interactions = len(YTrain)

            for drugIndex, diseaseIndex in nonInteractionIndices[trainNonInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTrain.append([0])

            nonInteractions = len(YTrain) - interactions
            print('train: interactions: ', interactions, 'non-interactions: ', nonInteractions)

            allDataDic[FEATURE_LIST[featureIndex]] = np.array(XTrain)
        
        YTrain = np.array(YTrain)
        allDataDic['diseases'] = np.array(involvedDiseases)
        allDataDic['labels'] = YTrain
        trainedModel = trainFNN(allDataDic, EMBEDDING_DEM, EPOCHS, BATCHSIZE, DROPOUT, 
            LEARNING_RATE, FEATURE_LIST, AGGREGATE_METHOD)

        # TESTING
        for featureIndex in range(len(FEATURE_LIST)):
            XTest = []
            YTest = []
            involvedDiseases = []
            for drugIndex, diseaseIndex in interactionIndices[testInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTest.append([1])

            interactions = len(YTest)

            for drugIndex, diseaseIndex in nonInteractionIndices[testNonInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTest.append([0])

            nonInteractions = len(YTest) - interactions
            print('test: interactions: ', interactions, 'non-interactions: ', nonInteractions)

            allDataDic[FEATURE_LIST[featureIndex]] = np.array(XTest)

        YTest = np.array(YTest)
        allDataDic['diseases'] = np.array(involvedDiseases)
    
        y_pred_prob = testFNN(trainedModel, allDataDic, FEATURE_LIST, AGGREGATE_METHOD).detach().numpy()

        metric = np.array(calculateMetric(YTest, y_pred_prob))
        metrics += metric
        print('metric: ', metric)

    return metrics

def main():
    print('nDrugs: ', DRUG_NUMBER)
    print('nDisease: ', DISEASE_NUMBER)
    
    drugDic = prepareData(FEATURE_LIST, EMBEDDING_METHOD)

    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
    interactionIndices = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    nonInteractionIndices = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
    
    results = crossValidation(drugDic, diseaseSim, drugDisease, interactionIndices, nonInteractionIndices)
    print('results: ', results / FOLDS)

main()
