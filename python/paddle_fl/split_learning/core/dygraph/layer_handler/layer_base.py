import paddle.fluid as fluid
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class LayerBase(fluid.dygraph.Layer):

    def __init__(self):
        super(LayerBase, self).__init__()

    def forward(self, inputs):
        """ return loss """
        raise NotImplementedError("Failed to run forward")

    def get_fetch_vars(self):
        raise NotImplementedError("Failed to get fetch vars")
