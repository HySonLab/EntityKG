import torch
import wandb

from pyvi import ViTokenizer
from torch_geometric.nn import VGAE
from core.loader import GraphDataset, GraphDataLoader
from core.model import GATEncoder, SuperGATEncoder

from utils import read_config, get_nodes_and_edges

# Training function
def training(model, data_loader, optimizer, num_epochs=200, eval_interval=10):
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(model, data):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edge_index)
            auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        return auc, ap

    for epoch in range(1, num_epochs + 1):
        for data in data_loader:
            loss = train(model, data, optimizer)
        
        if epoch % eval_interval == 0:
            auc, ap = evaluate(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

    # Final evaluation
    auc, ap = evaluate(model, data)
    print(f'Final AUC: {auc:.4f}, AP: {ap:.4f}')
    return auc, ap


####################################################################################################
config = read_config('./config/config.yaml')
nodes, edges = get_nodes_and_edges(config['data'])
nodes = [ViTokenizer.tokenize(node) for node in nodes]
edges = [(ViTokenizer.tokenize(edge[0]), ViTokenizer.tokenize(edge[1])) for edge in edges]

model_id = config['model']['encoder']
batch_size = config['model']['batch_size']

# Create the dataset
dataset = GraphDataset(nodes[:1000], edges[:500], model_id, split_kwargs=config['split_kwargs'])

# Create the data loader
data_loader = GraphDataLoader(dataset, batch_size=batch_size).get_loader()

# Create the VGAE model with GATEncoder
out_channels = config['model']['out_channels']
model = VGAE(GATEncoder(dataset.data.num_features, out_channels))

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

# Train the model
num_epochs = config['model']['num_epochs']
eval_interval = config['model']['eval_interval']

# Train the model
auc, ap = training(model, data_loader, optimizer, num_epochs, eval_interval)
