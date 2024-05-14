import sys
sys.path.insert(0, '../')

from core.loader import GraphDataset, GraphDataLoader

# Example usage:
nodes = [0, "biến chứng", 1, "não", "đột quỵ", "cục máu đông", 2, "tắc các mạch máu não", 3, "tim", "tắc mạch"]
edges = [[0, "biến chứng"], [1, "não"], [1, "đột quỵ"], [1, "cục máu đông"], [2, "tắc các mạch máu não"],
         [2, "biến chứng"], [2, "cục máu đông"], [3, "cục máu đông"], [3, "tim"], [3, "tắc mạch"]]

# Create the dataset
dataset = GraphDataset(nodes, edges)

# Create the data loader
data_loader = GraphDataLoader(dataset, batch_size=1).get_loader()