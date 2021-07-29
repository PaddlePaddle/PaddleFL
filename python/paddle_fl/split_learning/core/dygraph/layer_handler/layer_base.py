import paddle
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class LayerBase(paddle.nn.Layer):

    def __init__(self):
        super(LayerBase, self).__init__()

    @paddle.jit.to_static
    def forward(self, **feed):
        raise NotImplementedError("Failed to run forward")

    def get_fetch_vars(self):
        raise NotImplementedError("Failed to get fetch vars")

    def get_loss(self, inputs, predict):
        raise NotImplementedError("Failed to get loss")
