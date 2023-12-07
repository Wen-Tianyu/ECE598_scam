import pickle as pkl
import numpy as np
import torch
import csv
import os
import networkx as nx
import math

with open('./data/IDs.pkl', 'rb') as f:
    IDs = pkl.load(f)

with open('./data/contractIDs.pkl', 'rb') as f:
    contract_IDs = pkl.load(f)

edge_index = np.load('./data/edge_index.npy')
edge_attr = np.load('./data/edge_attr.npy')
node_attr = np.load('./data/Topo_feat.npy')

ID2newID = {}
newID2ID = {}
cur_newID = 0
for ID in IDs:
    ID2newID[ID] = cur_newID
    newID2ID[cur_newID] = ID
    cur_newID += 1
with open('./data/newid2id.pkl', 'wb') as f:
    pkl.dump(newID2ID, f)

for i in range(edge_index.shape[-1]):
    edge_index[0, i] = ID2newID[edge_index[0, i]]
    edge_index[1, i] = ID2newID[edge_index[1, i]]

for i in range(len(contract_IDs)):
    contract_IDs[i] = ID2newID[contract_IDs[i]]

edges = {}
for i in range(edge_index.shape[-1]):
    src, dst = edge_index[0, i], edge_index[1, i]
    if src in contract_IDs and dst not in contract_IDs:
        if src not in edges:
            edges[src] = []
        edges[src].append(dst)
    if dst in contract_IDs and src not in contract_IDs:
        if dst not in edges:
            edges[dst] = []
        edges[dst].append(src)

mu = 0.7
for contract_ID in contract_IDs:
    if contract_ID in edges:
        nb_IDs = edges[contract_ID]
        aggr_nb_attr = np.mean([node_attr[nb] for nb in nb_IDs])
        node_attr[contract_ID] = mu * node_attr[contract_ID] + (1 - mu) * aggr_nb_attr

newID2finalID = {}
finalID2newID = {}
cur_finalID = 0
for contract_ID in contract_IDs:
    newID2finalID[contract_ID] = cur_finalID
    finalID2newID[cur_finalID] = contract_ID
    cur_finalID += 1
with open('./data/finalid2newid.pkl', 'wb') as f:
    pkl.dump(finalID2newID, f)

contract_edge_index = []
for i in range(cur_finalID):
    for j in range(i + 1, cur_finalID):
        edge_set_i = set(edges[finalID2newID[i]]) if finalID2newID[i] in edges else set([])
        edge_set_j = set(edges[finalID2newID[j]]) if finalID2newID[j] in edges else set([])
        if edge_set_i.intersection(edge_set_j):
            contract_edge_index.append([i, j])
            contract_edge_index.append([j, i])
contract_edge_index = np.array(contract_edge_index).T
np.save('./data/contract_edge_index.npy', contract_edge_index)

final_node_attr = np.array([node_attr[finalID2newID[i]] for i in finalID2newID])

with open('./data/scamIDs.pkl', 'rb') as f:
    scam_IDs = pkl.load(f)
final_scam_IDs = [newID2finalID[ID2newID[scam_ID]] for scam_ID in scam_IDs]
y = np.zeros(final_node_attr.shape[0])
for final_scam_ID in final_scam_IDs:
    y[final_scam_ID] = 1

np.save('./data/final/x.npy', final_node_attr)
np.save('./data/final/y.npy', y)

print('Finished.')
