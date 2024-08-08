import torch

from typing import Tuple, List
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data
from utils import Node

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            node_attributes: List[Node], 
            edges: List[Tuple[str, str]],
            embedding_id: str
        ):
        # Store the node attributes and edges
        self.node_attributes = node_attributes
        self.edges = edges

        # Create the node to index and index to node mappings
        self.node_to_index = {node.text: i for i, node in enumerate(node_attributes)}
        self.index_to_node = {i: node for node, i in self.node_to_index.items()}

        # Store unique labels for the nodes
        self.labels = list(set([node.node_type for node in node_attributes]))

        # Create the data object
        if embedding_id == "random":
            self.tokenizer = None
            self.encoder = None
            self.data = self.create_data_with_random_features()
        else:
            # Load the transformer model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_id)
            self.encoder = AutoModel.from_pretrained(embedding_id).to(device)
            self.data = self.create_data()

    def create_data(self):
        # Create the edge index
        edge_index = torch.tensor([[self.node_to_index[edge[0]], self.node_to_index[edge[1]]] for edge in self.edges], dtype=torch.long).t()

        # Create the node features
        x = [self.tokenizer.encode(node.text) for node in self.node_attributes]
        x = [torch.tensor([e]) for e in x] # Add a dimension for batch
        x = torch.tensor([self.encoder(e).last_hidden_state.mean(dim=1)[0].tolist() for e in x])

        # Create the node labels
        y = [self.labels.index(node.node_type) for node in self.node_attributes]
        y = torch.tensor(y, dtype=torch.long)
        
        # Create the data object
        data = Data(x=x, edge_index=edge_index, y=y, num_classes=len(self.labels))
        return data
    
    def create_data_with_random_features(self):
        # Create the edge index
        edge_index = torch.tensor([[self.node_to_index[edge[0]], self.node_to_index[edge[1]]] for edge in self.edges], dtype=torch.long).t()

        # Create the node features
        x = torch.randn(len(self.node_attributes), 768).to(device)

        # Create the node labels
        y = [self.labels.index(node.node_type) for node in self.node_attributes]
        y = torch.tensor(y, dtype=torch.long)

        # Calculate edge attributes by element-wise product of node embeddings with dim [num_edges, num_edge_features]
        edge_attr = x[edge_index[0]] * x[edge_index[1]]
        
        # Create the data object
        data = Data(x=x, edge_index=edge_index, y=y, num_classes=len(self.labels), edge_attr=edge_attr)
        return data

    def __len__(self):
        return 1 # Only one graph in the dataset

    def __getitem__(self):
        return self.data
