import logging
import paddle

from .layer_base import LayerBase

_LOGGER = logging.getLogger(__name__)


class HostLayerHandler(object):

    def __init__(self, layer, optimizer):
        if not isinstance(layer, LayerBase):
            raise TypeError(
                    "Failed: layer({}) must be subclass of LayerBase."
                    .format(type(layer)))
        self.layer = layer
        self.optimizer = optimizer
        
        self.fetch_vars = None # not thread safe, for backward

    def call_for_forward(self, **inputs):
        self.fetch_vars = self.layer(**inputs)
        if not isinstance(self.fetch_vars, (list, tuple)):
            self.fetch_vars = [self.fetch_vars]
        return self.fetch_vars

    def call_for_backward(self, grads, tensor_names_to_customer):
        temp_var = 0
        for idx, name in enumerate(tensor_names_to_customer):
            var = self.fetch_vars[idx]
            var_grad = grads["{}@GRAD".format(name)]
            temp_var += var * var_grad
        
        temp_var.backward()
        self.optimizer.step()
        self.layer.clear_gradients()

    def cancel(self):
        self.layer.clear_gradients()
        self.fetch_vars = None
