import paddle.fluid as fluid
import logging

from .layer_base import LayerBase

_LOGGER = logging.getLogger(__name__)


class CustomerLayerHandler(object):

    def __init__(self, layer, optimizer):
        if not isinstance(layer, LayerBase):
            raise TypeError(
                    "Failed: layer({}) must be subclass of LayerBase."
                    .format(type(layer)))
        self.layer = layer
        self.optimizer = optimizer
        self.loss = None # not thread safe, for backward
    
    def call_for_forward(self, inputs):
        self.loss = self.layer(inputs)
        self.loss.backward()

    def call_for_backward(self):
        self.optimizer.minimize(self.loss)
        self.layer.clear_gradients()

    def get_fetch_vars(self):
        # tensor is not trainable
        return self.layer.get_fetch_vars()

    def cancel(self):
        self.layer.clear_gradients()
        self.loss = None
