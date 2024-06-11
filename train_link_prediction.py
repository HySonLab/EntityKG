import wandb

from torch_geometric.nn import VGAE
from torch.optim import Adam
from pyvi import ViTokenizer
from torch_geometric.transforms import RandomLinkSplit

from utils import read_config, get_nodes_and_edges, Node, ModelLinkSwitcher
from core.loader import GraphDataset


# Create the trainer
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_label_index)
    loss.backward()
    optimizer.step()

    # Calculate the auc, ap
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return loss, auc, ap

def validate(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return auc, ap

def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    return auc, ap

# Read the configuration file
# Read the configuration file
config = read_config('./config/config.yaml')
nodes, edges = get_nodes_and_edges(config['data'])

# Sample the data
nodes = [Node(text=ViTokenizer.tokenize(node.text), node_type=node.node_type, features=node.features) for node in nodes[:400]]
edges = [(ViTokenizer.tokenize(edge[0]), ViTokenizer.tokenize(edge[1])) for edge in edges[:200]]

# Load config
embedding_id = config['model']['encoder']
batch_size = config['model']['batch_size']

# Create the dataset
dataset = GraphDataset(nodes, edges, embedding_id)
transform = RandomLinkSplit(**config["splitted_edge_kwargs"])
train_data, val_data, test_data = transform(dataset.data)

# Create the model
# Parameters
model_name = config['model']['model_name']
learning_rate = config['model']['learning_rate']
model = VGAE(ModelLinkSwitcher(model_name)(train_data.num_features, config['model']['out_channels']))
optimizer = Adam(model.parameters(), lr=learning_rate)

# Log in wandb
wandb.init(project='Entity Knowledge Graph', tags=['link_prediction'])

num_epochs = config['model']['num_epochs']
for epoch in range(1, num_epochs + 1):
    loss, train_auc, train_ap = train(model, train_data, optimizer)
    val_auc, val_ap = validate(model, val_data)
    test_auc, test_ap = test(model, test_data)

    wandb.log({
        'loss': loss,
        'train_auc': train_auc,
        'train_ap': train_ap,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'test_auc': test_auc,
        'test_ap': test_ap
    })
    
    if epoch % 10 == 0:
        print(
            f'Epoch: {epoch:03d}, Loss: {loss:.4f}, \
            AUC Train: {train_auc:.4f}, AP Train: {train_ap:.4f}, \
            AUC Val: {val_auc:.4f}, AP Val: {val_ap:.4f}, \
            AUC Test: {test_auc:.4f}, AP Test: {test_ap:.4f}'
        )

# Log model name
wandb.log({"Model": model_name})

# Log additional information
wandb.log({"Num epochs": num_epochs})
wandb.log({"Optimizer": optimizer.__class__.__name__})
wandb.log({"Learning rate": learning_rate})
wandb.log({"Embedding": config['model']['encoder']})
wandb.log({"Batch size": batch_size})
wandb.log({"Out channels": config['model']['out_channels']})
wandb.log({"split": config['splitted_edge_kwargs']})

# Finish wandb run
wandb.finish()
