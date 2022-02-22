from prepareData import makePosEdgeIndex
from torch_geometric.data import HeteroData
import numpy as np 
import torch

def dataloader(dataset):
    print('dataset: ', dataset)
    data = HeteroData()
    if dataset == 'deepDR':
        numbers = {
            'drug': 1519,
            'disease': 1229,
            'protein': 1025,
            'sideEffect': 12904,
            'interactions': 6677,
            'nonInteractions': 1860174,
        }
        drugDisease = np.loadtxt('./data/' + 'deepDR' + '/drug_disease.txt', '\t')
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(6677, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(1860174, 2)
        drugDrug = np.loadtxt('./data/' + 'deepDR' + '/drug_drug.txt', delimiter='\t')
        data['drug'].x = torch.tensor(drugDrug, dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['protein'].x = torch.eye(numbers.get('protein'), dtype=torch.float)
        data['sideEffect'].x = torch.eye(numbers.get('sideEffect'), dtype=torch.float)

        data['drug', 'edge', 'protein'].edge_index = makePosEdgeIndex(dataset, 'drug_protein.txt', '\t')
        data['drug', 'edge', 'sideEffect'].edge_index = makePosEdgeIndex(dataset, 'drug_sideeffect.txt', '\t')

    # if dataset == 'LAGCN':
    #     metaDic = {
    #         numbers: {
    #             'drug': 10,
    #             'disease': 300,
    #             'pathway': 40,
    #             'enzyme': 70,
    #             'target': 99
    #         },
    #         drugFeatureList: ['enzyme', 'pathway', 'structure', 'target'],
    #         delimiter: ','
    #     }


    # drugDisease = np.loadtxt('./data/' + dataset + '/drug_disease.csv', delimiter=metaDic.delimiter)
    # totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    # totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)

    # data['drug'].x = torch.eye(metaDic.number.drug, dtype=torch.float)
    # data['disease'].x = torch.eye(metaDic.number.disease, dtype=torch.float)

    # for featureName in metaDic.drugFeatureList:
    #     data[featureName].x = torch.eye(metaDic.numbers[featureName], dtype=torch.float)
    #     data['drug', 'edge', featureName].edge_index = makePosEdgeIndex(featureName)

    return data, totalInteractions, totalNonInteractions
