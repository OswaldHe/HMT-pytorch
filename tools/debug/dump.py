import torch
import pickle
import os


class DebugDumper:
    def __init__(self, dump_dir='debug_dumps'):
        global _global_storage
        if '_global_storage' not in globals():
            _global_storage = {}
        self.storage = _global_storage
        self.dump_dir = dump_dir

        if self.dump_dir and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

    def store(self, key, value, step = 0):
        """
        Store a value with a given key.
        """
        if key not in self.storage:
            self.storage[key] = {}

        self.storage[key][step] = value

    def store_no_step(self, key, value):
        """
        Store a value with a given key without a step.
        """
        if key not in self.storage:
            self.storage[key] = []

        self.storage[key].append(value)

    def clear(self):
        """
        Clear all stored data.
        """
        self.storage.clear()

    def dump_to_file(self, filename='debug_dumps.pt'):
        """
        Dump all stored objects to a Python file.
        """
        torch.save(self.storage, os.path.join(self.dump_dir, filename))

# Example usage:
# dumper = DebugDumper()
# dumper.store("variable1", "value1")
# dumper.store_multiple(variable2=42, variable3=[1, 2, 3])
# dumper.dump_to_file("debug_output.py")
