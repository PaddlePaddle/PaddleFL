

import paddle
import io
import math
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

class YoutubeDNN(object):
    def input_data(self, batch_size, watch_vec_size, search_vec_size, other_feat_size):
        watch_vec = pfl_mpc.data(name='watch_vec', shape=[batch_size, watch_vec_size], dtype='int64')
        search_vec = pfl_mpc.data(name='search_vec', shape=[batch_size, search_vec_size], dtype='int64')
        other_feat = pfl_mpc.data(name='other_feat', shape=[batch_size, other_feat_size], dtype='int64')
        label = pfl_mpc.data(name='label', shape=[batch_size, 3952], dtype='int64')

        inputs = [watch_vec] + [search_vec] + [other_feat] + [label]

        return inputs

    def fc(self, tag, data, out_dim, active='relu'):
        init_stddev = 1.0
        scales = 1.0 / np.sqrt(data.shape[2])

        mpc_one = 65536 / 3
        rng = np.random.RandomState(23355)
        param_shape = (1, data.shape[2], out_dim) # 256, 2304

        if tag == 'l4':
            param_value_float = rng.normal(loc=0.0, scale=init_stddev * scales, size=param_shape)
            param_value_float_expand = np.concatenate((param_value_float, param_value_float), axis=0)
            param_value = (param_value_float_expand * mpc_one).astype('int64')
            initializer_l4 = pfl_mpc.initializer.NumpyArrayInitializer(param_value)
            p_attr = fluid.param_attr.ParamAttr(name='%s_weight' % tag,
                                                initializer=initializer_l4)
            active = None
        else:
            """
            param_init=pfl_mpc.initializer.XavierInitializer(seed=23355)
            p_attr = fluid.param_attr.ParamAttr(initializer=param_init)
            """
            """
            param_init=pfl_mpc.initializer.XavierInitializer(uniform=False, seed=23355)
            p_attr = fluid.param_attr.ParamAttr(initializer=param_init)
            """
            fan_in = param_shape[1]
            fan_out = param_shape[2]
            scale = math.sqrt(6.0 / (fan_in + fan_out))
            param_value_float = rng.normal(-1.0 * scale, 1.0 * scale, size=param_shape)
            param_value_float_expand = np.concatenate((param_value_float, param_value_float), axis=0)
            param_value = (param_value_float_expand * mpc_one * scale).astype('int64')
            initializer_l4 = pfl_mpc.initializer.NumpyArrayInitializer(param_value)
            p_attr = fluid.param_attr.ParamAttr(name='%s_weight' % tag,
                                                initializer=initializer_l4)


        b_attr = fluid.ParamAttr(name='%s_bias' % tag,
                                 initializer=fluid.initializer.ConstantInitializer(int(0.1 * mpc_one)))

        out = pfl_mpc.layers.fc(input=data,
                                size=out_dim,
                                act=active,
                                param_attr=p_attr,
                                bias_attr=b_attr,
                                name=tag)
        return out

    def net(self, inputs, output_size, layers=[32, 32, 32]):
        concat_feats = fluid.layers.concat(input=inputs[:-1], axis=-1)

        l1 = self.fc('l1', concat_feats, layers[0], 'relu')
        l2 = self.fc('l2', l1, layers[1], 'relu')
        l3 = self.fc('l3', l2, layers[2], 'relu')
        l4 = self.fc('l4', l3, output_size, None)
        cost, softmax = pfl_mpc.layers.softmax_with_cross_entropy(logits=l4,
                                                         label=inputs[-1],
                                                         soft_label=True,
                                                         use_relu=True,
                                                         use_long_div=False,
                                                         return_softmax=True)
        avg_cost = pfl_mpc.layers.mean(cost)
        return avg_cost, l3
