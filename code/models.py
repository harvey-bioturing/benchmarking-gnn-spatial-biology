import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, APPNP, GatedGraphConv, TransformerConv, TAGConv, DNAConv, ChebConv

#from torch_geometric.nn import global_mean_pool, DenseDiffPool

class TAGNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, K=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TAGConv(in_channels, hidden_channels, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_channels, hidden_channels, K=K))
        self.convs.append(TAGConv(hidden_channels, out_channels, K=K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class TransformerGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class ResGatedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.conv = GatedGraphConv(out_channels=hidden_channels, num_layers=num_layers)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)
        return x



# ========= GNN Model Classes =========
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3, heads=1):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(nn1))
        for _ in range(num_layers - 2):
            nnx = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nnx))
        nnf = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.convs.append(GINConv(nnf))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3, K=10, alpha=0.1):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))
        self.linears.append(nn.Linear(hidden_channels, out_channels))
        self.prop = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for lin in self.linears[:-1]:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return self.prop(x, edge_index)


class RecurrentGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.lstm = LSTM(hidden_channels, hidden_channels, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.gcn_layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_seq = x.unsqueeze(1)  # shape: [N, 1, F]
        x_lstm, _ = self.lstm(x_seq)
        x = self.lin(x_lstm[:, -1, :])
        return x


class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, K=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K=K))
        self.convs.append(ChebConv(hidden_channels, out_channels, K=K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)



# ========= Model Selection Dict =========
MODEL_CLASSES = {
    'gcn': GCN,
    'sage': GraphSAGE,
    'gat': GAT,
    'gin': GIN,
    'appnp': APPNPNet,
    'recurrent': RecurrentGNN,
    'cheb': ChebNet,
    'tag': TAGNet,
    'transformer': TransformerGNN,
    'gated': ResGatedGNN,
}
