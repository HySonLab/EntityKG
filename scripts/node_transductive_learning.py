import argparse
import torch
import wandb
import os, sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from huggingface_hub import login
from torch_geometric.transforms import RandomNodeSplit
from core.metrics import auc_score, ap_score
from core.utils import ModelNodeSwitcher, read_config, get_nodes_and_edges
from core.loader import MultiGraphDataset


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

def test(data, model):
    model.eval()
    out = model(data.x, data.edge_index)

    # Get the AUC and AP scores for test
    out = out.argmax(dim=1)
    auc = auc_score(out, data.y, data.test_mask)
    ap = ap_score(out, data.y, data.test_mask)

    return auc, ap


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse arguments
    parser = argparse.ArgumentParser(description='Node Inductive Learning')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--embedding', type=str, required=True, help='Name of the embedding model')
    parser.add_argument('--type', type=str, default='Transductive', help='Type of the task')
    args = parser.parse_args()

    # Read the config file
    config = read_config(args.config)

    # Login to the HuggingFace Hub
    login(config["hf_token"])

    # Load the dataset
    train_nodes, train_edges = get_nodes_and_edges(config["train_data"])
    test_nodes, test_edges = get_nodes_and_edges(config["test_data"])

    embedding_id = args.embedding
    # dataset = MultiGraphDataset(train_nodes, train_edges, test_nodes, test_edges, embedding_id)
    dataset = MultiGraphDataset(train_nodes[:400], train_edges[:200], test_nodes[:400], test_edges[:200], embedding_id)
    data = dataset.data.to(device)

    # Create the model
    model_names = config["train"]["model_names"]
    learning_rate = config["train"]["learning_rate"]
    for model_name in model_names:
        model = ModelNodeSwitcher(model_name)(data.num_features, data.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config["train"]["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss()

        # Log in wandb
        wandb.init(project=config["train"]["project_name"], tags=['node-classification'])

        num_epochs = config["train"]["num_epochs"]
        for epoch in range(1, num_epochs + 1):
            loss, train_auc, train_ap = train(data, model, criterion, optimizer)
            test_auc, test_ap = test(data, model)

            wandb.log({
                'loss': loss,
                'train_auc': train_auc,
                'train_ap': train_ap,
                'test_auc': test_auc,
                'test_ap': test_ap
            })
            
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch:03d}, Loss: {loss:.4f}, \
                    AUC Train: {train_auc:.4f}, AP Train: {train_ap:.4f}, \
                    AUC Test: {test_auc:.4f}, AP Test: {test_ap:.4f}'
                )

        # Log model name
        wandb.log({"Model": model_name})
        wandb.log({"Type": args.type})

        # Log additional information
        wandb.log({"Num epochs": num_epochs})
        wandb.log({"Optimizer": optimizer.__class__.__name__})
        wandb.log({"Learning rate": learning_rate})
        wandb.log({"Embedding": embedding_id})
        wandb.log({"Out channels": data.num_classes})

        # Finish wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
