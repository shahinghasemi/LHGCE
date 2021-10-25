import torch
from prepareData import prepareDrugData
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
INTERACTIONS_NUMBER = 18416
NONINTERACTIONS_NUMBER = 142446

def crossValidation(drugDic, diseaseSim, totalInteractions, totalNonInteractions):
    metrics = np.zeros(7)

    sizeOfInteractions = totalInteractions.shape[0]
    sizeOfNonInteractions = totalNonInteractions.shape[0]

    totalInteractionIndices = np.arange(sizeOfInteractions)
    totalNonInteractionIndices = np.arange(sizeOfNonInteractions)

    np.random.shuffle(totalInteractionIndices)
    np.random.shuffle(totalNonInteractionIndices)

    interactionsIndicesFolds = totalInteractionIndices.reshape(FOLDS, sizeOfInteractions // FOLDS)
    nonInteractionsIndicesFolds = totalNonInteractionIndices.reshape(FOLDS, sizeOfNonInteractions // FOLDS)

    for k in range(FOLDS):
        testInteractionsIndex = interactionsIndicesFolds[k]
        trainInteractionsIndex = np.setdiff1d(interactionsIndicesFolds.flatten(), testInteractionsIndex, assume_unique=True)

        testNonInteractionsIndex = nonInteractionsIndicesFolds[k]
        trainNonInteractionsIndex = np.setdiff1d(nonInteractionsIndicesFolds.flatten(), testNonInteractionsIndex, assume_unique=True)


        allDataDic = {}
        for featureIndex in range(len(FEATURE_LIST)):
            involvedDiseases = []
            XTrain = []
            YTrain = []
            for drugIndex, diseaseIndex in totalNonInteractions[trainNonInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTrain.append([0])
            
            # we won't use non interactions in training phase
            noninteractions = len(YTrain)
            print('train: noninteractions: ', noninteractions)

            allDataDic[FEATURE_LIST[featureIndex]] = np.array(XTrain)
        
        YTrain = np.array(YTrain)
        allDataDic['diseases'] = np.array(involvedDiseases)
        allDataDic['labels'] = YTrain
        trainedModel = trainFNN(allDataDic, EPOCHS, BATCHSIZE, DROPOUT, 
            LEARNING_RATE, FEATURE_LIST, AGGREGATE_METHOD)

        # TESTING
        for featureIndex in range(len(FEATURE_LIST)):
            XTest = []
            YTest = []
            involvedDiseases = []
            for drugIndex, diseaseIndex in totalInteractions:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTest.append([1])

            interactions = len(YTest)

            for drugIndex, diseaseIndex in totalNonInteractions[testNonInteractionsIndex]:
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
    
    drugDic = prepareDrugData(FEATURE_LIST, EMBEDDING_METHOD)
    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')

    totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
    
    # make it dividable by 5
    totalInteractions = totalInteractions[0:18415,:]
    totalNonInteractions = totalNonInteractions[0: 142445, :]
    # we want to have 50% of the nonInteractions to select 10% of them in each fold 
    # -3 is to make it dividable by 5
    # selectionSize = round(NONINTERACTIONS_NUMBER * 5/10) - 3
    # selection = np.random.choice(NONINTERACTIONS_NUMBER, selectionSize)
    # totalNonInteractions = totalNonInteractions[selection]

    results = crossValidation(drugDic, diseaseSim, totalInteractions, totalNonInteractions)
    print('results: ', results / FOLDS)

main()
