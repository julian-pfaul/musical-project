import torch
import torch.nn

class Model(torch.nn.Module):
    def __init__(self, identifier):
        super().__init__()

        self.meta_data = {
                'identifier': identifier
        }
