import torch

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import train_test_split_edges

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, nodes, edges, model_id, split_kwargs={}):
        self.nodes = nodes
        self.edges = edges
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id)
        self.split_kwargs = split_kwargs
        self.node_to_index = {node: i for i, node in enumerate(nodes)}
        self.index_to_node = {i: node for node, i in self.node_to_index.items()}
        self.data = self.create_data()

    def create_data(self):
        edge_index = torch.tensor([[self.node_to_index[edge[0]], self.node_to_index[edge[1]]] for edge in self.edges], dtype=torch.long).t()
        x = [self.tokenizer.encode(node) for node in self.nodes]
        x = [torch.tensor([e]) for e in x] # Add a dimension for batch
        x = torch.tensor([self.encoder(e).last_hidden_state.mean(dim=1)[0].tolist() for e in x])
        data = Data(x=x, edge_index=edge_index)
        data = train_test_split_edges(data, **self.split_kwargs) # type: ignore
        return data

    def __len__(self):
        return 1  # Only one graph in this dataset

    def __getitem__(self, idx):
        return self.data

class GraphDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_loader(self):
        return self.loader
