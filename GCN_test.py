import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ImbalancedSampler

from GNN_layer import GCN


def generate_mask(y, train_ratio):
    pos_idx = (y == 1).nonzero().view(-1).tolist()
    neg_idx = (y == 0).nonzero().view(-1).tolist()

    pos_permute = np.random.permutation(pos_idx)
    neg_permute = np.random.permutation(neg_idx)
    train_idx = np.concatenate([pos_permute[:int(train_ratio * len(pos_permute))],
                                neg_permute[:int(train_ratio * len(neg_permute))]])
    test_idx = np.concatenate([pos_permute[int(train_ratio * len(pos_permute)):],
                               neg_permute[int(train_ratio * len(neg_permute)):]])

    return train_idx.tolist(), test_idx.tolist()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.tensor(np.load('./data/final/x.npy'), dtype=torch.float32)
y = torch.tensor(np.load('./data/final/y.npy'), dtype=torch.long)
edge_index = torch.tensor(np.load('./data/contract_edge_index.npy'), dtype=torch.long)
data = Data(x=x, y=y, edge_index=edge_index)

train_mask, test_mask = generate_mask(y, 0.9)
train_sampler = ImbalancedSampler(data, input_nodes=train_mask)
test_sampler = ImbalancedSampler(data, input_nodes=test_mask)
train_loader = NeighborLoader(data, input_nodes=train_mask, batch_size=64, num_neighbors=[-1, -1], sampler=train_sampler)
test_loader = NeighborLoader(data, input_nodes=test_mask, batch_size=64, num_neighbors=[-1, -1], sampler=test_sampler)

x = (x - x.mean(dim=0)) / x.std(dim=0)

hidden_dim = 64
num_layers = 6
drop_out = 0.2
lr = 1e-3
wd = 1e-5
epochs = 100
model = GCN(x.size(dim=-1), hidden_dim, int(y.max().item()) + 1, num_layers, drop_out).to(device)
x = x.to(device)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 100], dtype=torch.float32))

train_losses = []
train_f1s = []
test_f1s = []

for epoch in range(epochs):
    train_loss_epoch = []
    train_f1_epoch = []
    test_f1_epoch = []

    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index)
        loss_t = loss(output, batch.y)
        loss_t.backward()
        optimizer.step()
        train_loss = loss_t.detach()
        train_loss_epoch.append(train_loss)
        pred = output.argmax(dim=1)
        train_acc = (pred == batch.y).sum() / pred.shape[0]
        train_f1 = f1_score(batch.y, pred, average='macro')
        train_f1_epoch.append(train_f1)
        train_infos = {
            'Epoch': epoch,
            'TrainLoss': '{:.3}'.format(train_loss.item()),
            'TrainAcc': '{:.3}'.format(train_acc.item()),
            'TrainF1': '{:.3}'.format(train_f1)
        }
        print(train_infos)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch.x, batch.edge_index)
            pred = output.argmax(dim=1)
            test_acc = (pred == batch.y).sum() / pred.shape[0]
            test_f1 = f1_score(batch.y, pred, average='macro')
            test_f1_epoch.append(test_f1)
            test_infos = {
                'Epoch': epoch,
                'TestAcc': '{:.3}'.format(test_acc.item()),
                'TestF1': '{:.3}'.format(test_f1)
            }
            print(test_infos)
    train_losses.append(np.mean(train_loss_epoch))
    train_f1s.append(np.mean(train_f1_epoch))
    test_f1s.append(np.mean(test_f1_epoch))

# Test on original imbalanced data
with torch.no_grad():
    scam_prob = model(x, edge_index)[:, 1].view(-1)
    noscam_prob = model(x, edge_index)[:, 0].view(-1)
    k = 20
    pred_scam_ID = torch.argsort(scam_prob, descending=True)[:k].tolist()
    pred_noscam_ID = torch.argsort(scam_prob, descending=True)[:k].tolist()
    with open('./data/finalid2newid.pkl', 'rb') as f:
        finalID2newID = pkl.load(f)
    with open('./data/newid2id.pkl', 'rb') as f:
        newID2ID = pkl.load(f)
    pred_scam_ID = [newID2ID[finalID2newID[finalID]] for finalID in pred_scam_ID]
    pred_noscam_ID = [newID2ID[finalID2newID[finalID]] for finalID in pred_noscam_ID]
    with open('./data/id2addr.pkl', 'rb') as f:
        ID2addr = pkl.load(f)
    pred_scam_addr = [ID2addr[id] for id in pred_scam_ID]
    pred_noscam_addr = [ID2addr[id] for id in pred_noscam_ID]

    print('Scams results.')
    for addr in pred_scam_addr:
        print(addr)

    print('Noscams results.')
    for addr in pred_noscam_addr:
        print(addr)

plt.plot(np.arange(len(train_losses)), train_losses)
plt.title('train_loss')
plt.show()
plt.plot(np.arange(len(train_f1s)), train_f1s)
plt.title('train_f1')
plt.show()
plt.plot(np.arange(len(test_f1s)), test_f1s)
plt.title('test_f1')
plt.show()
