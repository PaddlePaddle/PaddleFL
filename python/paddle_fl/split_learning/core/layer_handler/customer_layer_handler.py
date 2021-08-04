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
    
    def call_for_forward(self, label, **inputs):
        fetch_vars = self.layer(**inputs)
        if not isinstance(fetch_vars, (list, tuple)):
            fetch_vars = [fetch_vars]
        loss = self.layer.get_loss(fetch_vars, label)
        loss.backward()
        return fetch_vars

    def call_for_backward(self):
        self.optimizer.step()
        self.layer.clear_gradients()

    def cancel(self):
        self.layer.clear_gradients()
