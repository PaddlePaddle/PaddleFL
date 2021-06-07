#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import unittest
import warnings
import numpy as np
import random
import six
import time
import itertools
import collections
from collections import defaultdict
from multiprocessing import Pipe, Process, Manager
import os
import traceback
import unittest
import redis

import paddle_fl.mpc as pfl_mpc
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle_fl.mpc.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder, Variable
from testsuite import create_op, set_input, append_input_output, append_loss_ops
from paddle.fluid import unique_name
import traceback
from paddle_fl.mpc.data_utils.data_utils import get_datautils
import mpc_data_utils as mdu
from paddle_fl.mpc.framework import MpcProtocols


aby3 = get_datautils('aby3')


def _set_use_system_allocator(value=None):
    USE_SYSTEM_ALLOCATOR_FLAG = "FLAGS_use_system_allocator"
    old_value = core.globals()[USE_SYSTEM_ALLOCATOR_FLAG]
    value = old_value if value is None else value
    core.globals()[USE_SYSTEM_ALLOCATOR_FLAG] = value
    return old_value


def randomize_probability(batch_size, class_num, dtype='float32'):
    prob = np.random.uniform(
        0.1, 1.0, size=(batch_size, class_num)).astype(dtype)
    prob_sum = prob.sum(axis=1)
    for i in six.moves.xrange(len(prob)):
        prob[i] /= prob_sum[i]
    return prob

def skip_check_grad_ci(reason=None):
    """Decorator to skip check_grad CI.

       Check_grad is required for Op test cases. However, there are some special
       cases that do not need to do check_grad. This decorator is used to skip the
       check_grad of the above cases.

       Note: the execution of unit test will not be skipped. It just avoids check_grad
       checking in tearDownClass method by setting a `no_need_check_grad` flag.

       Example:
           @skip_check_grad_ci(reason="For inference, check_grad is not required.")
           class TestInference(OpTest):
    """
    if not isinstance(reason, str):
        raise AssertionError("The reason for skipping check_grad is required.")

    def wrapper(cls):
        cls.no_need_check_grad = True
        return cls

    return wrapper

class Aby3Process(Process):
    """
    Extends from Process, evaluate the computation party in aby3.
    """
    def __init__(self, *args, **kwargs):
        Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = Pipe()
        self._exception = None

    def run(self):
        """
        Override. Send any exceptions raised in
        subprocess to main process.
        """
        try:
            Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        """
        Get exception.
        """
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

class OpTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(OpTest, self).__init__(methodName)
        # set redis server and port
        self.server = os.environ['TEST_REDIS_IP']
        self.port = os.environ['TEST_REDIS_PORT']
        self.party_num = 3

    def setUp(self):
        """
        Connect redis and delete all keys in all databases on the current host.
        :return:
        """
        r = redis.Redis(host=self.server, port=int(self.port))
        r.flushall()

    def lazy_reconstruct(self, shares):
        return np.array(shares[0] / mdu.aby3_one_share).astype('float')

    def lazy_share(self, plain):
        return np.array([plain * mdu.aby3_one_share] * 2).astype('int64')

    def multi_party_run(self, **kwargs):
        """
        Run 3 parties with target function or other additional arguments.
        :param kwargs:
        :return:
        """
        r = redis.Redis(host=self.server, port=int(self.port))
        r.flushall()

        target = kwargs['target']

        parties = []

        for role in range(self.party_num):
            kwargs.update({'role': role})
            parties.append(Aby3Process(target=target, kwargs=kwargs))
            parties[-1].start()
        for party in parties:
            party.join()
            if party.exception:
                return party.exception
        return (True,)

    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()
        cls.call_once = False
        cls.dtype = None
        cls.outputs = {}
        cls.input_shape_is_large = True

        np.random.seed(123)
        random.seed(124)

        cls._use_system_allocator = _set_use_system_allocator(True)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        _set_use_system_allocator(cls._use_system_allocator)

        def is_empty_grad_op(op_type):
            all_op_kernels = core._get_all_register_op_kernels()
            grad_op = op_type + '_grad'
            if grad_op in all_op_kernels.keys():
                return False
            return True

        if not hasattr(cls, "op_type"):
            raise AssertionError(
                "This test do not have op_type in class attrs, "
                "please set self.__class__.op_type=the_real_op_type manually.")

        if not hasattr(cls, "no_need_check_grad") \
            and not is_empty_grad_op(cls.op_type):

            if not cls.input_shape_is_large and not hasattr(cls, "exist_check_grad"):
                raise AssertionError(
                    "Input's shape should be large than or equal to 100 for " +
                    cls.op_type + " Op.")

    def try_call_once(self, data_type):
        if not self.call_once:
            self.call_once = True
            self.dtype = data_type

    def infer_dtype_from_inputs_outputs(self, inputs, outputs):
        def is_np_data(input):
            return isinstance(input, (np.ndarray, np.generic))

        def infer_dtype(numpy_dict, dtype_set):
            assert isinstance(
                numpy_dict,
                dict), "self.inputs, self.outputs must be numpy_dict"
            # the inputs are as follows:
            # case 1: inputs = {'X': x}
            # case 2: inputs = {'X': (x, x_lod)}
            # case 3: inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
            # case 4: inputs = {'X': [("x1", (x1, [x1_lod1])), ("x2", (x2, [x2_.lod2]))]}
            # TODO(juncaipeng) infer dtype from inputs maybe obtain wrong type.
            for _, var_value in six.iteritems(numpy_dict):
                if is_np_data(var_value):  # case 1
                    dtype_set.add(var_value.dtype)
                elif isinstance(var_value, (list, tuple)):  # case 2, 3, 4
                    for sub_val_value in var_value:
                        if is_np_data(sub_val_value):  # case 2
                            dtype_set.add(sub_val_value.dtype)
                        elif len(sub_val_value) > 1 and is_np_data(
                                sub_val_value[1]):  # case 3
                            dtype_set.add(sub_val_value[1].dtype)
                        elif len(sub_val_value) > 1 and isinstance(sub_val_value[1], (list, tuple)) \
                            and is_np_data(sub_val_value[1][0]): # case 4
                            dtype_set.add(sub_val_value[1][0].dtype)

        # infer dtype from inputs, and dtype means the precision of the test
        # collect dtype of all inputs
        dtype_set = set()
        infer_dtype(inputs, dtype_set)
        dtype_list = [
            np.dtype(np.float64), np.dtype(np.float32), np.dtype(np.float16),
            np.dtype(np.int64), np.dtype(np.int32), np.dtype(np.int16),
            np.dtype(np.int8), np.dtype(np.uint8), np.dtype(np.bool)
        ]
        # check the dtype in dtype_list in order, select the first dtype that in dtype_set
        for dtype in dtype_list:
            if dtype in dtype_set:
                self.dtype = dtype
                break
        # save dtype in class attr
        self.__class__.dtype = self.dtype

    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], place)
                        tensor.set_recursive_sequence_lengths(np_value[1])
                    else:
                        tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    tensor.set_recursive_sequence_lengths(self.inputs[var_name][
                        1])
                else:
                    tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def _append_ops(self, block):
        self.__class__.op_type = self.op_type  # for ci check, please not delete it for now

        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
        "infer datatype from inputs and outputs for this test case"
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        inputs = append_input_output(block, op_proto, self.inputs, True,
                                     self.dtype)
        outputs = append_input_output(block, op_proto, self.outputs, False,
                                      self.dtype)

        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=self.attrs if hasattr(self, "attrs") else dict())
        # infer variable type and infer shape in compile-time
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        return op

    def _get_io_vars(self, block, numpy_inputs):
        inputs = {}
        for name, value in six.iteritems(numpy_inputs):
            if isinstance(value, list):
                var_list = [
                    block.var(sub_name) for sub_name, sub_value in value
                ]
                inputs[name] = var_list
            else:
                inputs[name] = block.var(name)
        return inputs

    def _get_inputs(self, block):
        return self._get_io_vars(block, self.inputs)

    def _get_outputs(self, block):
        return self._get_io_vars(block, self.outputs)

    def calc_output(self, place):
        outs, _ = self._calc_output(place)
        return outs

    def _calc_output(self,
                     place,
                     parallel=False,
                     no_check_set=None,
                     loss=None,
                     enable_inplace=None,
                     for_inplace_test=False):
        program = Program()
        block = program.global_block()
        op = self._append_ops(block)

        inputs = self._get_inputs(block)
        outputs = self._get_outputs(block)
        feed_map = self.feed_var(inputs, place)

        if for_inplace_test:
            # Some variables' tensors hold no buffer (tensor's _holder is NULL), like XShape in reshape2 op,
            # and the shapes of those variables contain 0 (eg. Xshape.shape = [0, 2, 5]).
            # Set persistable for those variables in order to get them from global_scope for inplace grad test directly other than feed them,
            # since feed op calls check_memory_size() which fails when tensor's holder_ is NULL.
            for name in op.output_arg_names:
                var = block.var(name)
                var.persistable = True
        original_program = program
        #if parallel:
        #    use_cuda = False
        #    if isinstance(place, fluid.CUDAPlace):
        #        use_cuda = True
        #    compiled_prog = fluid.CompiledProgram(program).with_data_parallel(
        #        loss_name=loss.name if loss else None, places=place)
        #    program = compiled_prog
        fetch_list = getattr(self, "fetch_list", [])
        # if the fetch_list is customized by user, we use it directly.
        # if not, fill the fetch_list by the user configured outputs in test.
        if len(fetch_list) == 0:
            for var_name, var in six.iteritems(outputs):
                if no_check_set is not None and var_name in no_check_set:
                    continue
                if isinstance(var, list):
                    for v in var:
                        fetch_list.append(v.name)
                else:
                    fetch_list.append(var.name)
        # if the fetch_list still empty, fill the fetch_list by the operator output.
        if len(fetch_list) == 0:
            for out_name, out_dup in Operator.get_op_outputs(self.op_type):
                fetch_list.append(str(out_name))

        if enable_inplace is not None:
            build_strategy = fluid.BuildStrategy()
            build_strategy.enable_inplace = enable_inplace

            compiled_prog = fluid.CompiledProgram(program).with_data_parallel(
                build_strategy=build_strategy, places=place)
            program = compiled_prog
        # Manager() can not store LoDTensor directly
        # So, use one additional element to store output lod
        return_results = [Manager().list() for _ in range(len(fetch_list) + 1)]

        def closure(**kwargs):
            role = kwargs['role']

            pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))

            #init_op = fluid.default_main_program().global_block().ops[0]

            #_insert_init_op(program, init_op)

            executor = Executor(place)

            executor.run()
            outs = executor.run(program,
                            feed=feed_map,
                            fetch_list=fetch_list,
                            return_numpy=False)
            lod = []
            for idx in range(len(fetch_list)):
                return_results[idx].append(np.array(outs[idx]))

                lod_i = outs[idx].lod()
                lod_concat = []
                for i in lod_i:
                    lod_concat.append(i)
                lod.append(lod_concat)
            return_results[len(fetch_list)].append(lod)

        ret = self.multi_party_run(target=closure)
        self.assertEqual(ret[0], True)

        outs = []
        lod = np.array(return_results[len(fetch_list)])
        for idx in range(len(fetch_list)):
            t = fluid.LoDTensor()
            reveal_data = aby3.reconstruct(np.array(return_results[idx]))
            t.set(reveal_data, place)
            lod_idx = lod[0][idx]
            
            try:
                t.set_lod(lod_idx)
            except Exception as e:
                pass

            outs.append(t)

        self.op = op
        self.program = original_program
        if for_inplace_test:
            return outs, fetch_list, feed_map, original_program, op.desc
        else:
            return outs, fetch_list

    def _get_need_run_ops(self, op_desc, fwd_op_desc=None):
        """Postorder traversal of the 'grad' tree to get all ops that need to run during inplace test.
        An op needs to run druing inplace check if,
        (1) it has infer_inplace,
        (2) it has infer_inplace in its grad descendants. (since we need its outputs as to construct its grad's inputs)

        Args:
            op_desc (OpDesc): The op_desc of current op.
            fwd_op_desc (OpDesc): The op_desc of current op's forward op, None if current op has no forward op.
                Eg. relu's fwd_op is None, relu_grad's fwd_op is relu, relu_grad_grad's fwd_op is relu_grad, etc.

        Returns:
            need_run_ops (list[(op_desc, fwd_op_desc)]): The ops that need to run during inplace test.
        """
        need_run_ops = []
        visited_ops = []

        def _dfs_grad_op(op_desc, fwd_op_desc=None):
            visited_ops.append(op_desc.type())
            has_infer_inplace = fluid.core.has_infer_inplace(op_desc.type())
            has_grad_op_maker = fluid.core.has_grad_op_maker(op_desc.type())
            has_infer_inplace_in_grad_descendants = False
            if not has_grad_op_maker:
                has_infer_inplace_in_descendants = False
            else:
                # get grad_op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    op_desc, set(), [])
                if not grad_op_desc_list:
                    has_infer_inplace_in_grad_descendants = False
                else:
                    for i, grad_op_desc in enumerate(grad_op_desc_list):
                        if grad_op_desc.type(
                        ) not in visited_ops and _dfs_grad_op(
                                grad_op_desc, fwd_op_desc=op_desc):
                            has_infer_inplace_in_grad_descendants = True
            if has_infer_inplace or has_infer_inplace_in_grad_descendants:
                need_run_ops.append((op_desc, fwd_op_desc))
                return True
            else:
                return False

        _dfs_grad_op(op_desc, fwd_op_desc=fwd_op_desc)
        return need_run_ops

    def check_inplace_output_with_place(self,
                                        place,
                                        no_check_set=None,
                                        inplace_atol=None):
        """Chech the inplace correctness of given op, its grad op, its grad_grad op, etc.

        (1) Get all ops need to run. (see conditions in _get_need_run_ops())
        (2) Run op in need_run_ops, and do inplace check if it has infer_inplace.

        Args:
            place (CPUPlace | CUDAPlace): The place where the op runs.
            no_check_set (list): The names of outputs that needn't check, like XShape of reshape op.
            inplace_atol (float): The tolerable error, only set when op doesn't ensure computational consistency, like group_norm op.

        Returns:
            None
        """
        has_infer_inplace = fluid.core.has_infer_inplace(self.op_type)
        has_grad_op_maker = fluid.core.has_grad_op_maker(self.op_type)

        fwd_res = self._calc_output(
            place, no_check_set=no_check_set, for_inplace_test=True)
        op_desc = fwd_res[4]
        need_run_ops = self._get_need_run_ops(op_desc)

        res = {}
        for op_desc, father_op_desc in reversed(need_run_ops):
            # The first one is the forward op
            has_infer_inplace = fluid.core.has_infer_inplace(op_desc.type())
            if op_desc.type() == self.op_type:
                if has_infer_inplace:
                    res[op_desc] = self._check_forward_inplace(
                        place,
                        no_check_set=no_check_set,
                        inplace_atol=inplace_atol)
                else:
                    res[op_desc] = self._calc_output(
                        place, no_check_set=no_check_set, for_inplace_test=True)
            else:
                if has_infer_inplace:
                    fwd_res = res[father_op_desc]
                    res[op_desc] = self._check_grad_inplace(
                        place, fwd_res, op_desc, inplace_atol=inplace_atol)
                else:
                    res[op_desc] = self._calc_grad_output(place, fwd_res,
                                                          op_desc)

    def check_output_with_place(self,
                                place,
                                atol=0,
                                no_check_set=None,
                                equal_nan=False,
                                check_dygraph=True,
                                inplace_atol=None):
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)

        prog = Program()
        block = prog.global_block()
        # wirte protocol name and index into global scope
        op_init = block.append_op(
            type="mpc_init",
            #inputs=inputs,
            #outputs=outputs,
            attrs={"protocol_name": "aby3"})
        op_init.desc.infer_shape(block.desc)

        mpc_protocol_index = MpcProtocols["ABY3"].value
        fluid.global_scope().var("mpc_protocol_index").get_tensor().set(
            np.array((mpc_protocol_index)), fluid.CPUPlace())

        outs, fetch_list = self._calc_output(place, no_check_set=no_check_set)
        for out_name, out_dup in Operator.get_op_outputs(self.op_type):
            if out_name not in self.outputs:
                continue
            if no_check_set is not None and out_name in no_check_set:
                continue

            def find_imperative_actual(target_name, dygraph_outs, place):
                with fluid.dygraph.base.guard(place=place):
                    for name in dygraph_outs:
                        if name == target_name:
                            return dygraph_outs[name][0]
                        var_list = dygraph_outs[name]
                        for i, var in enumerate(var_list):
                            if var.name == target_name:
                                return dygraph_outs[name][i]
                    self.assertTrue(False, "Found failed {} {}".format(
                        dygraph_outs.keys(), target_name))

            def find_actual(target_name, fetch_list):
                found = [
                    i for i, var_name in enumerate(fetch_list)
                    if var_name == target_name
                ]
                self.assertTrue(
                    len(found) == 1, "Found {} {}".format(
                        len(found), target_name))
                return found[0]

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))
                for item in sub_out:
                    sub_out_name, expect = item[0], item[1]
                    idx = find_actual(sub_out_name, fetch_list)
                    actual = outs[idx]
                    actual_t = np.array(actual)
                    expect_t = expect[0] \
                        if isinstance(expect, tuple) else expect
                    expect_t = self.lazy_reconstruct(expect_t)
                    self.assertTrue(
                        np.allclose(
                            actual_t, expect_t, atol=atol, equal_nan=equal_nan),
                        "Output (" + sub_out_name + ") has diff at " +
                        str(place))
                    if isinstance(expect, tuple):
                        self.assertListEqual(
                            actual.recursive_sequence_lengths(), expect[1],
                            "Output (" + sub_out_name +
                            ") has different lod at " + str(place))
            else:
                idx = find_actual(out_name, fetch_list)
                actual = outs[idx]
                actual_t = np.array(actual)
                expect = self.outputs[out_name]
                expect_t = expect[0] if isinstance(expect, tuple) else expect
                expect_t = self.lazy_reconstruct(expect_t)
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, atol=atol, equal_nan=equal_nan),
                    "Output (" + out_name + ") has diff at " + str(place) +
                    "\nExpect " + str(expect_t) + "\n" + "But Got" +
                    str(actual_t) + " in class " + self.__class__.__name__)
                if isinstance(expect, tuple):
                    self.assertListEqual(actual.recursive_sequence_lengths(),
                                         expect[1], "Output (" + out_name +
                                         ") has different lod at " + str(place))

        # Note(zhiqiu): inplace_atol should be only set when op doesn't ensure
        # computational consistency.
        # For example, group_norm uses AtomicAdd on CUDAPlace, which do not ensure
        # computation order when multiple threads write the same address. So the
        # result of group_norm is non-deterministic when datatype is float.
        # When inplace_atol is not None, the inplace check uses numpy.allclose
        # to check inplace result instead of numpy.array_equal.
        if inplace_atol is not None:
            warnings.warn(
                "inplace_atol should only be set when op doesn't ensure computational consistency, please check it!"
            )
        # Check inplace for given op, its grad op, its grad_grad op, etc.
        # No effect on original OpTest
        self.check_inplace_output_with_place(
            place, no_check_set=no_check_set, inplace_atol=inplace_atol)

        return outs, fetch_list

    def _assert_is_close(self, numeric_grads, analytic_grads, names,
                         max_relative_error, msg_prefix):

        for a, b, name in six.moves.zip(numeric_grads, analytic_grads, names):
            # It asserts np.abs(a - b) / np.abs(a) < max_relative_error, in which
            # max_relative_error is 1e-7. According to the value of np.abs(a), we
            # change np.abs(a) to achieve dynamic threshold. For example, if
            # the value of np.abs(a) is between 1e-10 and 1e-8, we set np.abs(a)*=1e4.
            # Therefore, it asserts np.abs(a - b) / (np.abs(a)*1e4) < max_relative_error,
            # which is the same as np.abs(a - b) / np.abs(a) < max_relative_error*1e4.
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return ("%s error, %s variable %s max gradient diff %f over limit %f, "
                    "the first error element is %d, expected %f, but got %f.") \
                    % (self.op_type, msg_prefix, name, max_diff, max_relative_error,
                    offset, a.flatten()[offset], np.array(b).flatten()[offset])

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def _check_grad_helper(self):
        self.infer_dtype_from_inputs_outputs(self.inputs, self.outputs)
        self.__class__.op_type = self.op_type
        self.__class__.exist_check_grad = True

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None,
                              check_dygraph=True,
                              transpose_input_list=[]):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()

        prog = Program()
        block = prog.global_block()
        # wirte protocol name and index into global scope
        op_init = block.append_op(
            type="mpc_init",
            #inputs=inputs,
            #outputs=outputs,
            attrs={"protocol_name": "aby3"})
        op_init.desc.infer_shape(block.desc)

        mpc_protocol_index = MpcProtocols["ABY3"].value
        fluid.global_scope().var("mpc_protocol_index").get_tensor().set(
            np.array((mpc_protocol_index)), fluid.CPUPlace())

        self._check_grad_helper()

        cache_list = None
        if hasattr(self, "cache_name_list"):
            cache_list = self.cache_name_list
        self.op = create_op(
            self.scope,
            self.op_type,
            op_inputs,
            op_outputs,
            op_attrs,
            cache_list=cache_list)

        if no_grad_set is None:
            no_grad_set = set()

        for input_to_check in inputs_to_check:
            set_input(self.scope, self.op, self.inputs, place)
            tensor_to_check = self.scope.find_var(input_to_check).get_tensor()
            tensor_size = six.moves.reduce(lambda a, b: a * b,
                                           tensor_to_check.shape(), 1)
            if tensor_size < 100:
                self.__class__.input_shape_is_large = False

        if not type(output_names) is list:
            output_names = [output_names]

        numeric_grads = user_defined_grads or [
            self.get_numeric_gradient(
                place,
                self.scope,
                self.op,
                self.inputs,
                input_to_check,
                output_names,
                delta=numeric_grad_delta,
                in_place=in_place,
                transpose_input_list=transpose_input_list) for input_to_check in inputs_to_check
        ]
        analytic_grads = self._get_gradient(inputs_to_check, place,
                                            output_names, no_grad_set)
        self._assert_is_close(numeric_grads, analytic_grads, inputs_to_check,
                              max_relative_error,
                              "Gradient Check On %s" % str(place))



    @staticmethod
    def _numpy_to_lod_tensor(np_value, lod, place):
        tensor = core.LoDTensor()
        tensor.set(np_value, place)
        if lod is not None:
            tensor.set_recursive_sequence_lengths(lod)
        return tensor

    @staticmethod
    def np_dtype_to_fluid_dtype(input):
        return input

    @staticmethod
    def fluid_dtype_to_np_dtype(self, dtype):
        return dtype

    @staticmethod
    def np_value_to_fluid_value(input):
        return input

    def _get_gradient(self,
                      input_to_check,
                      place,
                      output_names,
                      no_grad_set,
                      parallel=False):
        prog = Program()
        block = prog.global_block()
        self._append_ops(block)

        # Manager() can not store LoDTensor directly
        # So, use one additional element to store output lod
        fetch_list = []
        fetch_list_len = len(input_to_check)
        return_results = [Manager().list() for _ in range(fetch_list_len + 1)]

        def closure(**kwargs):
            role = kwargs['role']

            pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
            loss = append_loss_ops(block, output_names)
            param_grad_list = append_backward(
                loss=loss, parameter_list=input_to_check, no_grad_set=no_grad_set)

            inputs = self._get_inputs(block)
            feed_dict = self.feed_var(inputs, place)

            fetch_list = [g for p, g in param_grad_list]

            executor = Executor(place)

            executor.run()
            outs = executor.run(prog,
                            feed=feed_dict,
                            fetch_list=fetch_list,
                            return_numpy=False)
            # append lod information in last position
            lod = []
            for idx in range(fetch_list_len):
                return_results[idx].append(np.array(outs[idx]))
                lod_i = outs[idx].lod()
                lod_concat = []
                for i in lod_i:
                    lod_concat.append(i)
                lod.append(lod_concat)
            return_results[fetch_list_len].append(lod)

        ret = self.multi_party_run(target=closure)
        self.assertEqual(ret[0], True)

        outs = []

        lod = np.array(return_results[fetch_list_len])
        # from numpy array to LoDTensor
        for idx in range(fetch_list_len):
            t = fluid.LoDTensor()
            reveal_data = aby3.reconstruct(np.array(return_results[idx]))
            t.set(reveal_data, place)
            lod_idx = lod[0][idx]
            # TODO: fix: exception throw because some output lod error in gru op
            # out.set_lod(out.lod()) will throw exception
            try:
                t.set_lod(lod_idx)
            except Exception as e:
                pass

            outs.append(t)
        return outs

    def get_numeric_gradient(self,
                             place,
                             scope,
                             op,
                             inputs,
                             input_to_check,
                             output_names,
                             delta=0.005,
                             in_place=False,
                             transpose_input_list=[]):
        # FIXME: change this method by compile time concepts
        set_input(scope, op, inputs, place)

        def product(dim):
            return six.moves.reduce(lambda a, b: a * b, dim, 1)

        delta_origin = delta
        tensor_to_check = scope.find_var(input_to_check).get_tensor()
        input_shape = tensor_to_check.shape()
        
        # some input with lod need to transpose at begin
        # it is be reconstructed here first
        if input_to_check in transpose_input_list:
            tensor_to_check = np.transpose(tensor_to_check, [1, 0, 2])
        tensor_to_check_  = fluid.LoDTensor()
        tensor_to_check_.set(tensor_to_check, fluid.CPUPlace())
        tensor_to_check = tensor_to_check_

        tensor_size = int(product(input_shape) / 2)
        tensor_to_check_dtype = tensor_to_check._dtype()
        input_plain_shape = tensor_to_check.shape()[1:]
        if tensor_to_check_dtype == core.VarDesc.VarType.FP32:
            tensor_to_check_dtype = np.float32
        elif tensor_to_check_dtype == core.VarDesc.VarType.FP64:
            tensor_to_check_dtype = np.float64
        elif tensor_to_check_dtype == core.VarDesc.VarType.FP16:
            tensor_to_check_dtype = np.float16
            # set delta as np.float16, will automatic convert to float32, float64
            delta = np.array(delta).astype(np.float16)
        elif tensor_to_check_dtype == core.VarDesc.VarType.INT64:
            tensor_to_check_dtype = np.int64
            delta = np.array(delta * 2**16 / 3).astype(np.int64)
        else:
            raise ValueError("Not supported data type " + str(
                tensor_to_check_dtype))

        def get_output():
            sum = []

            return_results = dict()

            for name in (output_names):
                return_results[name] = Manager().list()

            def closure(**kwargs):
                role = kwargs['role']

                pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))

                executor = Executor(place)

                executor.run()
                op.run(scope, place)

                for name in output_names:
                    out  = np.array(scope.find_var(name).get_tensor())
                    return_results[name].append(out)

            ret = self.multi_party_run(target=closure)
            self.assertEqual(ret[0], True)

            for output_name in output_names:
                plain = aby3.reconstruct(np.array(return_results[output_name]))
                sum.append(plain.mean())

            return (np.array(sum).sum() / len(output_names)).astype(np.float)

        gradient_flat = np.zeros(shape=(tensor_size, ), dtype = np.float)

        def __get_elem__(tensor, i):
            if tensor_to_check_dtype == np.float16:
                numpy_tensor = np.array(tensor).astype(np.float16)
                numpy_tensor = numpy_tensor.flatten()
                return numpy_tensor[i]
            elif tensor_to_check_dtype == np.int64:
                numpy_tensor = np.array(tensor).astype(np.int64)
                numpy_tensor = numpy_tensor.flatten()
                return numpy_tensor[i]
            elif tensor_to_check_dtype == np.float32:
                return tensor._get_float_element(i)
            else:
                return tensor._get_double_element(i)

        def __set_elem__(tensor, i, e):
            if tensor_to_check_dtype == np.float16:
                numpy_tensor = np.array(tensor).astype(np.float16)
                shape = numpy_tensor.shape
                numpy_tensor = numpy_tensor.flatten()
                numpy_tensor[i] = e
                numpy_tensor = numpy_tensor.reshape(shape)
                tensor.set(numpy_tensor, place)
            elif tensor_to_check_dtype == np.int64:
                numpy_tensor = np.array(tensor).astype(np.int64)
                shape = numpy_tensor.shape
                numpy_tensor = numpy_tensor.flatten()
                numpy_tensor[i] = e
                numpy_tensor = numpy_tensor.reshape(shape)
                if input_to_check in transpose_input_list:
                    numpy_tensor = np.transpose(numpy_tensor, [1, 0, 2])
                    numpy_tensor = np.array(numpy_tensor).reshape(input_shape)
                tensor_input_origin = scope.find_var(input_to_check).get_tensor()
                tensor_input_origin.set(numpy_tensor, place)
            elif tensor_to_check_dtype == np.float32:
                tensor._set_float_element(i, e)
            else:
                tensor._set_double_element(i, e)

        # we only compute gradient of one element each time.
        # we use a for loop to compute the gradient of every element.
        for i in six.moves.xrange(tensor_size):
            if in_place:
                set_input(scope, op, inputs, place)

            # get one input element throw it's index i.
            origin = __get_elem__(tensor_to_check, i)
            origin1 = __get_elem__(tensor_to_check, tensor_size + i)
            # add delta to (shares0, shares1), run op and then get the sum of the result tensor.
            x_pos = origin + delta
            x_pos1 = origin1 + delta
            __set_elem__(tensor_to_check, i, x_pos)
            __set_elem__(tensor_to_check, tensor_size + i, x_pos1)
            y_pos = get_output()

            if in_place:
                set_input(scope, op, inputs, place)

            x_neg = origin - delta
            x_neg1 = origin1 - delta
            __set_elem__(tensor_to_check, i, x_neg)
            __set_elem__(tensor_to_check, tensor_size + i, x_neg1)
            y_neg = get_output()

            __set_elem__(tensor_to_check, i, origin)
            __set_elem__(tensor_to_check, tensor_size + i, origin1)
            gradient_flat[i] = (y_pos - y_neg) / delta_origin / 2

        return gradient_flat.reshape(input_plain_shape)
