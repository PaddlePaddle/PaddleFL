import paddle.fluid as fluid
import numpy as np
import grpc
import yaml

import data_iter

class MLP(fluid.dygraph.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.embed_dim = 11
        self.embed = fluid.dygraph.Embedding(
                size=[100, self.embed_dim],
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))
        self.pool = fluid.dygraph.Pool2D(
                pool_type='max',
                global_pooling=True)
        self.fc1 = fluid.dygraph.nn.Linear(
                input_dim=12, 
                output_dim=10,
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))
        self.fc2 = fluid.dygraph.nn.Linear(
                input_dim=10,
                output_dim=2,
                act='softmax',
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    def forward(self, inputs):
        self.embed_var = self.embed(inputs["x"])
        self.embed_var = fluid.layers.reshape(
                self.embed_var, 
                [-1, 12, 1, self.embed_dim])
        self.pool_var = self.pool(self.embed_var)
        self.pool_var = fluid.layers.reshape(
                self.pool_var, 
                [-1, 12])
        self.fc1_var = self.fc1(self.pool_var)
        print(self.fc1_var.numpy())
        self.fc2_var = self.fc2(self.fc1_var)
        return self.fc2_var


if __name__ == "__main__":
    place = fluid.CPUPlace()
    fluid.enable_imperative(place)
    model = MLP()

    optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=0.01, parameter_list=model.parameters())

    for epoch in range(5):
        print("======= epoch{} =======".format(epoch))
        for i, item in enumerate(data_iter.iter()):
            slot, label = item
            slot_var = fluid.dygraph.to_variable(slot)
            label_var = fluid.dygraph.to_variable(label)

            predict = model({"x": slot_var})
            print(predict)
            cost = fluid.layers.cross_entropy(predict, label_var)
            cost = fluid.layers.reduce_mean(cost)

            cost.backward()
            optimizer.minimize(cost)
            model.clear_gradients()
            break

    #fluid.save_dygraph(model.state_dict(), 'whole_model')    
