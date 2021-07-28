import paddle.fluid as fluid
import logging

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

    def call_for_forward(self, inputs):
        self.layer(inputs)
        self.fetch_vars = self.layer.get_fetch_vars()
        return self.fetch_vars

    def call_for_backward(self, grads):
        temp_var = 0
        for name, var in self.fetch_vars.items():
            var_grad = grads["{}@GRAD".format(name)]
            temp_var += var * var_grad
        
        temp_var.backward()
        self.optimizer.minimize(temp_var)
        self.layer.clear_gradients()

    def cancel(self):
        self.layer.clear_gradients()
        self.fetch_vars = None
