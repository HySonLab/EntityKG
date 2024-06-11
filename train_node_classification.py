import wandb

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from pyvi import ViTokenizer
from torch_geometric.transforms import RandomNodeSplit

from utils import read_config, get_nodes_and_edges, Node, ModelNodeSwitcher
from core.loader import GraphDataset
from core.metrics import auc_score, ap_score


# Train and evaluate the model
def train(data, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Get the AUC and AP scores for train
    out = out.argmax(dim=1)
    auc = auc_score(out, data.y, data.train_mask)
    ap = ap_score(out, data.y, data.train_mask)

    return loss, auc, ap

def evaluate(data, model):
    model.eval()
    out = model(data.x, data.edge_index)

    # Get the AUC and AP scores for test
    out = out.argmax(dim=1)
    auc = auc_score(out, data.y, data.val_mask)
    ap = ap_score(out, data.y, data.val_mask)

    return auc, ap

def test(data, model):
    model.eval()
    out = model(data.x, data.edge_index)

    # Get the AUC and AP scores for test
    out = out.argmax(dim=1)
    auc = auc_score(out, data.y, data.test_mask)
    ap = ap_score(out, data.y, data.test_mask)

    return auc, ap

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
transform = RandomNodeSplit(**config["splitted_node_kwargs"])
data = transform(dataset.data)

# Parameters
model_name = "GAT"
learning_rate = config['model']['learning_rate']
model = ModelNodeSwitcher(model_name)(data.num_features, data.num_classes)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()

# Log in wandb
wandb.init(project='Entity Knowledge Graph', tags=['node-classification'])

num_epochs = config['model']['num_epochs']
for epoch in range(1, num_epochs + 1):
    loss, train_auc, train_ap = train(data, model, criterion, optimizer)
    val_auc, val_ap = evaluate(data, model)
    test_auc, test_ap = test(data, model)

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
wandb.log({"Out channels": data.num_classes})
wandb.log({"split": config['splitted_node_kwargs']})

# Finish wandb run
wandb.finish()