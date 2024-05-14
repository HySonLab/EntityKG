import json
import yaml

def read_config(file_path):
    """Read and parse a YAML configuration file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration data.

    Raises:
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def get_nodes_and_edges(json_file):
    """Function to get nodes and edges from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    kg_nodes = data[0]
    kg_edges = data[1]
    
    return kg_nodes, kg_edges
