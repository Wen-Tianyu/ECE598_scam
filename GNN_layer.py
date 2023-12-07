import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, num_heads):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, self.num_heads, concat=False))
        for _ in range(self.num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, self.num_heads, concat=False))
        self.convs.append(GATConv(hidden_dim, out_dim, self.num_heads, concat=False))

        self.skips = nn.ModuleList()
        self.skips.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.skips.append(nn.Linear(hidden_dim, hidden_dim))
        self.skips.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = x + self.skips[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()

        self.lins = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim))
        self.convs.append(GINConv(self.lins[0]))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(self.lins[_ + 1]))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.convs.append(GINConv(self.lins[-1]))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
