import safetensors
from safetensors.torch import save_file

def add_format_to_metadata(path, format='pt'):
    """When loading the savetensor files using the from pretrained, "format is not supported" kind of error is reported. 
    This is because the 'format' field of the savetensor metadata is missing. 
    The script loads the desired safetensor file, set the metadata to pytorch, then write it back. 
    """
    
    tensors = dict()
    with safetensors.safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    save_file(tensors, path, metadata={'format': format})

