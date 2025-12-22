import torch


class SegmentIterator:
    def __init__(self, **kwargs):
        self.iter_content = kwargs
        self.pointer = 0
        self.empty = False

    def next(self, segment_length):
        segment = {}
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return None
                segment[k] = tensor[:, self.pointer : self.pointer + segment_length]

        self.pointer += segment_length
        return segment

    def is_empty(self):
        for _, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return True
                else:
                    return False
