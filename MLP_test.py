import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam


def generate_mask(length, train_ratio):
    train_mask = np.full(length, False)
    test_mask = np.full(length, False)

    permute = np.random.permutation(length)
    train_idx = permute[: int(train_ratio * length)]
    test_idx = permute[int(train_ratio * length):]
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return train_mask, test_mask


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.linear2(h)
        return F.log_softmax(h, dim=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_os = torch.tensor(np.load('./data/final/x_os.npy'), dtype=torch.float32)
y_os = torch.tensor(np.load('./data/final/y_os.npy'), dtype=torch.long)

x_os = (x_os - x_os.mean(dim=0)) / x_os.std(dim=0)

hidden_dim = 32
drop_out = 0.2
lr = 1e-3
wd = 1e-5
epochs = 5000
model = MLP(x_os.size(dim=-1), hidden_dim, int(y_os.max().item()) + 1, drop_out).to(device)
x_os = x_os.to(device)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
loss = nn.CrossEntropyLoss()

train_mask, test_mask = generate_mask(x_os.size(dim=0), 0.9)
train_losses = []
train_f1s = []
test_f1s = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_os)
    loss_t = loss(output[train_mask], y_os[train_mask])
    loss_t.backward()
    optimizer.step()
    train_loss = loss_t.detach()
    train_losses.append(train_loss)
    pred = output.argmax(dim=1)
    train_acc = (pred[train_mask] == y_os[train_mask]).sum() / train_mask.sum()
    train_f1 = f1_score(y_os[train_mask], pred[train_mask], average='macro')
    train_f1s.append(train_f1)
    model.eval()
    with torch.no_grad():
        output = model(x_os)
        val_loss = loss_t
        pred = output.argmax(dim=1)
        test_acc = (pred[test_mask] == y_os[test_mask]).sum() / test_mask.sum()
        test_f1 = f1_score(y_os[test_mask], pred[test_mask], average='macro')
        test_f1s.append(test_f1)
    infos = {
        'Epoch': epoch,
        'TrainLoss': '{:.3}'.format(train_loss.item()),
        'TrainAcc': '{:.3}'.format(train_acc.item()),
        'TrainF1': '{:.3}'.format(train_f1),
        'TestAcc': '{:.3}'.format(test_acc.item()),
        'TestF1': '{:.3}'.format(test_f1)
    }
    print(infos)


# Test on original imbalanced data
with torch.no_grad():
    x = torch.tensor(np.load('./data/final/x.npy'), dtype=torch.float32)
    x = (x - x.mean(dim=0)) / x.std(dim=0)

    scam_prob = model(x)[:, 1].view(-1)
    k = 30
    pred_scam_ID = torch.argsort(scam_prob, descending=True)[:k].tolist()
    with open('./data/finalid2newid.pkl', 'rb') as f:
        finalID2newID = pkl.load(f)
    with open('./data/newid2id.pkl', 'rb') as f:
        newID2ID = pkl.load(f)
    pred_scam_ID = [newID2ID[finalID2newID[finalID]] for finalID in pred_scam_ID]
    with open('./data/id2addr.pkl', 'rb') as f:
        ID2addr = pkl.load(f)
    pred_scam_addr = [ID2addr[id] for id in pred_scam_ID]
    for addr in pred_scam_addr:
        print(addr)

plt.plot(np.arange(len(train_losses)), train_losses)
plt.title('mlp_train_loss')
plt.show()
plt.plot(np.arange(len(train_f1s)), train_f1s)
plt.title('mlp_train_f1')
plt.show()
plt.plot(np.arange(len(test_f1s)), test_f1s)
plt.title('mlp_test_f1')
plt.show()

weight = model.linear1.weight.detach().numpy().T
plt.matshow(weight)
plt.colorbar()
plt.show()
