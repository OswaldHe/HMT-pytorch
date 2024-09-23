import json

def read_jsonl(file_path):
    """
    Read a jsonl file line by line and return a list of Python dictionaries.
    
    :param file_path: Path to the jsonl file
    :return: List of dictionaries, each representing a line from the jsonl file
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
