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
    
    def call_for_forward(self, inputs):
        loss = self.layer(inputs)
        loss.backward()

    def call_for_backward(self):
        self.optimizer.step()
        self.layer.clear_gradients()

    def get_fetch_vars(self):
        # tensor is not trainable
        return self.layer.get_fetch_vars()

    def cancel(self):
        self.layer.clear_gradients()
