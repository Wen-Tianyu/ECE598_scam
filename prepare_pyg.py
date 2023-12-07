import csv
import os
import pickle as pkl

import networkx as nx
import numpy as np

csv.field_size_limit(100000000)


def get_nx_digraph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    return G


path1 = './data/addr2id.pkl'
path2 = './data/trans_by_id.pkl'
path3 = './data/id2addr.pkl'
if not os.path.exists(path1) or not os.path.exists(path2) or not os.path.exists(path3):
    addr2id = dict()
    id2addr = dict()
    trans_by_id = []
    ID = 0
    head = True
    with open('./data/transactions.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if head:
                head = False
                continue
            if row:
                if row[3] not in addr2id:
                    addr2id[row[3]] = ID
                    id2addr[ID] = row[3]
                    ID += 1
                if row[4] not in addr2id:
                    addr2id[row[4]] = ID
                    id2addr[ID] = row[4]
                    ID += 1
                trans_by_id.append((addr2id[row[3]], addr2id[row[4]], float(row[5]) / 1e18))
    with open(path1, 'wb') as f:
        pkl.dump(addr2id, f)
    with open(path2, 'wb') as f:
        pkl.dump(trans_by_id, f)
    with open(path3, 'wb') as f:
        pkl.dump(id2addr, f)
else:
    with open(path1, 'rb') as f:
        addr2id = pkl.load(f)
    with open(path2, 'rb') as f:
        trans_by_id = pkl.load(f)
    with open(path3, 'rb') as f:
        id2addr = pkl.load(f)

g = get_nx_digraph(list(addr2id.values()), trans_by_id)

print(g.number_of_nodes())

# Remove nodes with degree less than 2
g.remove_nodes_from([node for node, degree in nx.degree(g) if degree < 2])
n = nx.number_of_nodes(g)
print(n)
dd = nx.degree(g)
nodes = list(nx.nodes(g))
path4 = './data/IDs.pkl'
if not os.path.exists(path4):
    with open(path4, 'wb') as f:
        pkl.dump(nodes, f)


contract_IDs = []
with open('./data/contracts_address_in_transactions.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        contract_addr = row[0]
        if contract_addr in addr2id:
            contract_id = addr2id[contract_addr]
            if contract_id in nodes:
                contract_IDs.append(contract_id)
path_cIDs = './data/contractIDs.pkl'
if not os.path.exists(path_cIDs):
    with open(path_cIDs, 'wb') as f:
        pkl.dump(contract_IDs, f)

scam_IDs = []
with open('./data/scams.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        scam_addr = row[0]
        if scam_addr in addr2id:
            scam_id = addr2id[scam_addr]
            if scam_id in nodes:
                scam_IDs.append(scam_id)
path_sIDs = './data/scamIDs.pkl'
if not os.path.exists(path_sIDs):
    with open(path_sIDs, 'wb') as f:
        pkl.dump(scam_IDs, f)

edge_index = []
edge_attr = []
for src, dst, w in g.edges(data='weight'):
    edge_index.append([src, dst])
    edge_attr.append([w])
path_edge_index = './data/edge_index.npy'
if not os.path.exists(path_edge_index):
    edge_index = np.array(edge_index).T
    np.save(path_edge_index, edge_index)
path_edge_attr = './data/edge_attr.npy'
if not os.path.exists(path_edge_attr):
    edge_attr = np.array(edge_attr)
    np.save(path_edge_attr, edge_attr)

# Degree Centrality
path5 = './data/C.npy'
if not os.path.exists(path5):
    dc_value = nx.degree_centrality(g)
    ic_value = nx.in_degree_centrality(g)
    oc_value = nx.out_degree_centrality(g)
    dc_value = np.array(list(dc_value.values())).reshape(-1, 1)
    ic_value = np.array(list(ic_value.values())).reshape(-1, 1)
    oc_value = np.array(list(oc_value.values())).reshape(-1, 1)
    C = np.hstack((dc_value, ic_value, oc_value))
    np.save(path5, C)

# PageRank and HITS
path6 = './data/P.npy'
if not os.path.exists(path6):
    pr_value = nx.pagerank(g)
    h_value, a_value = nx.hits(g)
    pr_value = np.array(list(pr_value.values())).reshape(-1, 1)
    h_value = np.array(list(h_value.values())).reshape(-1, 1)
    a_value = np.array(list(a_value.values())).reshape(-1, 1)
    P = np.hstack((pr_value, h_value, a_value))
    np.save(path6, P)

# Shortest path lengths
path7 = './data/D.npy'
if not os.path.exists(path7):
    D = np.full((n, n), -1)
    spl = dict(nx.all_pairs_shortest_path_length(g))
    for i, src in enumerate(nodes):
        if src not in spl:
            continue
        d = spl[src]
        for j, dst in enumerate(nodes):
            if dst not in d:
                continue
            D[i, j] = d[dst]
    np.save(path7, D)

path77 = './data/D6.npy'
if not os.path.exists(path77):
    D = np.load('./data/D.npy')
    in_1 = D.sum(axis=1).reshape(-1)
    out_1 = D.sum(axis=0).reshape(-1)
    in_2 = D.max(axis=1).reshape(-1)
    out_2 = D.max(axis=0).reshape(-1)
    in_3 = (D != -1).sum(axis=1).reshape(-1)
    out_3 = (D != -1).sum(axis=0).reshape(-1)
    D6 = np.transpose(np.stack((in_1, out_1, in_2, out_2, in_3, out_3)))
    np.save(path77, D6)


# Transaction values sum-up
path8 = './data/V.npy'
if not os.path.exists(path8):
    V = np.array(list(dict(nx.degree(g, weight='weight')).values())).reshape(-1, 1)
    np.save(path8, V)

# Concatenate all topology features into one matrix
path9 = './data/Topo_feat.npy'
if not os.path.exists(path9):
    C = np.load(path5)
    P = np.load(path6)
    D6 = np.load(path77)
    V = np.load(path8)
    Topo_feat = np.concatenate((C, P, D6, V), axis=1)
    np.save(path9, Topo_feat)
