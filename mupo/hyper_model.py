from .model import *
from .utils import *

import torch.nn

class HyperModel(Model):
    def __init__(self):
        super().__init__("hyper-model")
        
        # seq_key = str(seq_len) 
        # and seq_len is the accepted input sequence length of the model as an int
        #
        # special cases:
        # - the key "fallback" refers to a model used on sequences with arbitrary lengths
        #   in case a specialized model isn't present. 
        self.submodels = torch.nn.ModuleDict()


    def set_submodel(self, key, model):
        self.submodels[key] = model


    def set_sequence_model(self, seq_len, model):
        key = str(int(seq_len))

        self.submodels[key] = model

    def set_fallback_model(self, model):
        self.submodels["fallback"] = model
        

    def optimal_model_for_seq_len(self, seq_len):
        key = str(int(seq_len))

        if key in self.submodels.keys():
            return self.submodels[key]
        
        # we are trying to use the fallback model
        # in case there isn't a specialized model for seq_len
        
        key = "fallback"
        
        if key in self.submodels.keys():
            return self.submodels["fallback"]
        else:
            raise RuntimeError(
f"""
Couldn't find a specialized model for sequence length of {seq_len},
tried to resort to fallback model, which also couldn't be found.
Every instance of {self.__class__.__name__} has to have a fallback-model.

You have to register the fallback model with 
{self.__class__.__name__}.set_submodel("fallback", fallback_model)
or
{self.__class__.__name__}.set_fallback_model(fallback_model)

Additionally, you may also provide a specialized model for the sequence length of {seq_len} like so:
{self.__class__.__name__}.set_submodel(seq_len, model)
or
{self.__class__.__name__}.set_specialized_model(seq_len, model)
"""
            )


    def forward(self, x):
        seq_len = sequence_length(x)
        model = self.optimal_model_for_seq_len(seq_len)

        return model(x) # note: modules that are not used during this specific forward pass
                        # do not get their parameters updated during backpropagation.
