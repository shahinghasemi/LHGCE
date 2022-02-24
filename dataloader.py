from prepareData import makePosEdgeIndex
from torch_geometric.data import HeteroData
import numpy as np 
import torch
import pandas as pd

def dataloader(dataset):
    print('dataset: ', dataset)
    data = HeteroData()
    if dataset == 'LRSSL':
        numbers = {
            'drug': 763,
            'disease': 681,
            'go': 4447,
            'target': 1426,
            'structure': 623,
            'interactions': 3051,
            'nonInteractions': 516552,
        }

        drugDisease = pd.read_csv('./data/' + dataset + '/drug_dis_mat.txt', sep='\t', header=None).drop(columns=0, index=0).to_numpy(dtype=np.integer)
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(3051, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(516552, 2)
        
        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.tensor(pd.read_csv('./data/' + dataset + '/disease_similarity.txt', sep='\t', header=None).drop(columns=0, index=0).to_numpy(dtype=np.float64), dtype=torch.float)        
        data['go'].x = torch.eye(numbers.get('go'), dtype=torch.float)
        data['target'].x = torch.eye(numbers.get('target'), dtype=torch.float)
        data['structure'].x = torch.eye(numbers.get('structure'), dtype=torch.float)
      
        data['drug', 'edge', 'go'].edge_index = makePosEdgeIndex(dataset, 'drug_target_go_mat.txt', '\t', dataframe=True)
        data['drug', 'edge', 'target'].edge_index = makePosEdgeIndex(dataset, 'drug_target_domain_mat.txt', '\t', dataframe=True)
        data['drug', 'edge', 'structure'].edge_index = makePosEdgeIndex(dataset, 'drug_pubchem_mat.txt', '\t', dataframe=True)

    elif dataset == 'deepDR':
        numbers = {
            'drug': 1519,
            'disease': 1229,
            'protein': 1025,
            'sideEffect': 12904,
            'interactions': 6677,
            'nonInteractions': 1860174,
        }
        drugDisease = np.loadtxt('./data/' + 'deepDR' + '/drug_disease.txt', delimiter='\t')
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(6677, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(1860174, 2)

        data['drug'].x = torch.tensor(np.loadtxt('./data/' + 'deepDR' + '/drug_drug.txt', delimiter='\t'), dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['protein'].x = torch.eye(numbers.get('protein'), dtype=torch.float)
        data['sideEffect'].x = torch.eye(numbers.get('sideEffect'), dtype=torch.float)
        data['drug', 'edge', 'protein'].edge_index = makePosEdgeIndex(dataset, 'drug_protein.txt', '\t')
        # In order to prevent memory leak we only choose a proportion of the sideEffect network.
        data['drug', 'edge', 'sideEffect'].edge_index = makePosEdgeIndex(dataset, 'drug_sideeffect.txt', '\t', 50)

    elif dataset == 'LAGCN':
        numbers = {
            'drug': 269,
            'disease': 598,
            'pathway': 258,
            'enzyme': 108,
            'target': 529,
            'structure': 881,
            'interactions': 18416,
            'nonInteractions': 142446,
        }

        drugDisease = np.loadtxt('./data/' + 'LAGCN' + '/drug_disease.csv', delimiter=',')
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)

        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.tensor(np.loadtxt('./data/LAGCN/dis_sim.csv', delimiter=','), dtype=torch.float)        
        data['pathway'].x = torch.eye(numbers.get('pathway'), dtype=torch.float)
        data['enzyme'].x = torch.eye(numbers.get('enzyme'), dtype=torch.float)
        data['target'].x = torch.eye(numbers.get('target'), dtype=torch.float)
        data['structure'].x = torch.eye(numbers.get('structure'), dtype=torch.float)
      
        data['drug', 'edge', 'pathway'].edge_index = makePosEdgeIndex(dataset, 'drug_pathway.txt', ' ')
        data['drug', 'edge', 'enzyme'].edge_index = makePosEdgeIndex(dataset, 'drug_enzyme.txt', ' ')
        data['drug', 'edge', 'target'].edge_index = makePosEdgeIndex(dataset, 'drug_target.txt', ' ')
        data['drug', 'edge', 'structure'].edge_index = makePosEdgeIndex(dataset, 'drug_structure.txt', ' ')


    return data, totalInteractions, totalNonInteractions
