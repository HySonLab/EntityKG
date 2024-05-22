import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SuperGATConv


# SAGE for link prediction
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# GCN for link prediction
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# GAT for link prediction
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, heads=heads)
        self.conv_mu = GATConv(2 * out_channels * heads, out_channels, heads=1)
        self.conv_logstd = GATConv(2 * out_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# SuperGAT for link prediction
class SuperGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(SuperGATEncoder, self).__init__()
        self.conv1 = SuperGATConv(in_channels, 2 * out_channels, heads=heads)
        self.conv_mu = SuperGATConv(2 * out_channels * heads, out_channels, heads=1)
        self.conv_logstd = SuperGATConv(2 * out_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
