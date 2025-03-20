from .hyper_model import *
from .fallback_model import *
from .sequence_model import *

class AlphaModel(HyperModel):
    def __init__(self):
        super().__init__()

        fallback = FallbackModel()

        self.set_fallback_model(fallback)
