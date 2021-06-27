# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
This module provide data utilities for PrivC protocol, including
data encryption, decryption, share save and loading.
"""

import abc
import six
import os
import numpy as np
import six
import paddle
import paddle.fluid as fluid
import mpc_data_utils as mdu
from ..layers import __all__ as all_ops
from .op_extra_desc import add_extra_desc

# operators that should be skipped when encrypt and decrypt
op_to_skip = ['feed', 'fetch', 'scale', 'mpc_init']
# operators that are supported currently for model encryption and decryption
supported_mpc_ops = all_ops + ['fill_constant', 'sgd'] + op_to_skip
# variables that used as plain variables and need no encryption
plain_vars = ['learning_rate_0']

MPC_MODEL_NAME = "__model__.mpc"
MODEL_NAME = "__model__"
MODEL_SHARE_DIR = "model_share"
MPC_OP_PREFIX = "mpc_"

@six.add_metaclass(abc.ABCMeta)
class DataUtils(object):
    """
    abstract class for data utils.
    """

    def __init__(self):
        self.SHARE_NUM = None
        self.PRE_SHAPE = None
        self.MPC_ONE_SHARE = None

    def encrypt(self, number):
        """
        Encrypts the plaintext number into secret shares
        Args:
            number: float, the number to share
        Returns:
            shares of input number
        """
        pass

    def decrypt(self, shares):
        """
        Reveal plaintext value from raw secret shares
        Args:
            shares: shares to reveal from (list)
        Return:
           the plaintext number (float)
        """
        pass

    
    def make_shares(self, num_array):
        """
        Create raw shares for an array.

        Args:
            num_array: the input data array
        Returns:
            shares of the num_array in type of ndarray
        """
        old_size = num_array.size
        flat_num_array = num_array.reshape(old_size,)
        new_shape = (self.SHARE_NUM, ) + num_array.shape
        result = np.empty((old_size, self.SHARE_NUM), dtype=np.int64)
        for idx in six.moves.range(0, old_size):
            result[idx] = self.encrypt(flat_num_array[idx])

        result = result.transpose(1, 0)
        result = result.reshape(new_shape)
        return result
  

    def get_shares(self, shares, index):
        """
        Build mpc shares from raw shares according to index

        Args:
            shares: the input raw share array
            index: index of the mpc share
        Returns:
            mpc shares array corresponding to index
        """
        pass
 
 
    def save_shares(self, share_reader, part_name):
        """
        Combine raw shares to mpc shares, and persists to files. Each mpc share will be
        put into the corresponding file, e.g., ${part_name}.part[0/1/2].
        Args:
            share_reader: iteratable function object returning a single array of raw shares
                in shape of [2/3, ...] each time
            part_name: file name
        Returns:
            files with names of ${part_name}.part[0/1/2]
        """
        pass


    def load_shares(self, part_name, id, shape, append_share_dim=True):
        """
        Load mpc shares from file with name ${part_name}.part{id} in shape of ${shape}.

        Args:
            part_name and id: use to build the file name of ${part_name}.part{id}
            shape: the shape of output array
        Returns:
            iteratable function object returing a share array with give shape each time
        """
        if append_share_dim == True:
            shape = self.PRE_SHAPE + shape

        ext = ".part{}".format(id)
        share_size = np.prod(shape) * 8  # size of int64 in bytes

        def reader():
            """
            internal reader
            """
            with open(part_name + ext, 'rb') as part_file:
                share = part_file.read(share_size)
                while share:
                    yield np.frombuffer(share, dtype=np.int64).reshape(shape)
                    share = part_file.read(share_size)

        return reader


    def reconstruct(self, shares, type=np.float):
        """
        Reconstruct plaintext from mpc shares

        Args:
            shares: all the mpc share arrays, where the share slices
                    are stored rowwise
            type: expected type of final result
        Returns:
            plaintext array reconstructed from the mpc shares, with shape of (dims)
        """
        pass


    def batch(self, reader, batch_size, drop_last=False):
        """
        A batch reader return a batch data meeting the shared data's shape.
        E.g., a batch arrays with shape (3, 4) of batch_size will be transform to (batch_size, 3, 4).

        Args: see paddle.batch method
        Returns: the batched reader
        """
        pass


    def transpile(self, program=None):
        """
        Transpile Paddle program into MPC program.

        Args:
            program: The plain Paddle model program, default to
            default_main_program.

        Returns: The MPC program.
        """
        if program is None:
            program = fluid.default_main_program()

        place = fluid.CPUPlace()
        if program.num_blocks > 1:
            raise NotImplementedError(
                "The number of blocks in current main program"
                "is {}, which is not supported in this version."
                .format(program.num_blocks()))

        global_block = program.global_block()
        g_scope = fluid.global_scope()

        mpc_vars_names = _transpile_type_and_shape(block=global_block)

        # encrypt tensor values for each variable in mpc_var_names
        for mpc_var_name in mpc_vars_names:
            if g_scope.find_var(mpc_var_name) is not None:
                param = g_scope.find_var(mpc_var_name)
                param_tensor = np.array(param.get_tensor())
                mpc_var = global_block.var(mpc_var_name)
                if mpc_var_name not in plain_vars:
                    param.get_tensor()._set_dims(mpc_var.shape)
                    # process initialized params that should be 0
                    set_tensor_value = np.array([param_tensor, param_tensor]).astype(np.int64)
                    param.get_tensor().set(set_tensor_value, place)
                #else:
                #    param.get_tensor().set(np.array(param.get_tensor()).astype('float64'), place)

        # trigger sync to replace old ops.
        op_num = global_block.desc.op_size()
        _ = global_block.desc.append_op()
        global_block.desc._remove_op(op_num, op_num + 1)
        return program


    def _transpile_type_and_shape(self, block):
        """
        Transpile dtype and shape of plain variables into MPC dtype and shape.
        And transpile op type into MPC type.

        Args:
            block: The block in Paddle program.

        Returns: A set of variable names to encrypt.
        """
        mpc_vars_names = set()

        # store variable name in mpc_vars_names, and encrypt dtype and shape
        for var_name in block.vars:
            var = block.var(var_name)
            if var.name != "feed" and var.name != "fetch":
                mpc_vars_names.add(var.name)
                if var_name in plain_vars:
                    # var.desc.set_dtype(fluid.framework.convert_np_dtype_to_dtype_(np.float64))
                    continue
                encrypted_var_shape = self.PRE_SHAPE + var.shape
                var.desc.set_dtype(fluid.framework.convert_np_dtype_to_dtype_(np.int64))
                var.desc.set_shape(encrypted_var_shape)

        # encrypt op type, or other attrs if needed
        for op in block.ops:
            if _is_supported_op(op.type):
                if op.type == 'fill_constant':
                    op._set_attr(name='shape', val=MPC_ONE_SHARE.shape)
                    # set default MPC value for fill_constant OP
                    op._set_attr(name='value', val=MPC_ONE_SHARE)
                    op._set_attr(name='dtype', val=3)
                elif op.type in self.op_to_skip:
                    pass
                else:
                    add_extra_desc(op, block)
                    op.desc.set_type(MPC_OP_PREFIX + op.type)
            else:
                raise NotImplementedError('Operator {} is unsupported.'
                                          .format(op.type))
        return mpc_vars_names


    def encrypt_model(self, program, mpc_model_dir=None, model_filename=None):
        """
        Encrypt model, and save encrypted model (i.e., MPC model shares) into
        files for MPC training, updating, or inference.

        Args:
            program: The loaded program of paddle model.
            mpc_model_dir: The directory that save MPC model shares.
            model_filename: The name of MPC model file, default is __model__.mpc.
        """
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        # TODO(xukun): support more blocks. Tips: may just adding "for loop" for all blocks.
        if program.num_blocks > 1:
            raise NotImplementedError(
                "The number of blocks in current main program"
                "is {}, which is not supported in this version."
                .format(program.num_blocks()))

        global_block = program.global_block()
        g_scope = fluid.global_scope()

        mpc_vars_names = _transpile_type_and_shape(global_block)

        # encrypt tensor values for each variable in mpc_var_names
        for mpc_var_name in mpc_vars_names:
            if g_scope.find_var(mpc_var_name) is not None:
                param = g_scope.find_var(mpc_var_name)
                param_tensor = np.array(param.get_tensor())
                param_tensor_shares = self.make_shares(param_tensor)
                mpc_var = global_block.var(mpc_var_name)
                for idx in six.moves.range(self.SHARE_NUM):
                    if mpc_var_name not in plain_vars:
                        param.get_tensor()._set_dims(mpc_var.shape)
                        set_tensor_value = self.get_shares(param_tensor_shares, idx)
                        param.get_tensor().set(set_tensor_value, place)
                    #else:
                    #    param.get_tensor().set(np.array(param.get_tensor()).astype('float64'), place)

                    param_share_dir = os.path.join(
                        mpc_model_dir, MODEL_SHARE_DIR + "_" + str(idx))
                    fluid.io.save_vars(
                        executor=exe,
                        dirname=param_share_dir,
                        vars=[mpc_var],
                        filename=mpc_var_name)

        # trigger sync to replace old ops.
        op_num = global_block.desc.op_size()
        _ = global_block.desc.append_op()
        global_block.desc._remove_op(op_num, op_num + 1)

        # save mpc model file
        model_basename = os.path.basename(
            model_filename) if model_filename is not None else MPC_MODEL_NAME
        for idx in six.moves.range(self.SHARE_NUM):
            model_share_dir = os.path.join(mpc_model_dir,
                                           MODEL_SHARE_DIR + "_" + str(idx))
            if not os.path.exists(model_share_dir):
                os.makedirs(model_share_dir)
            model_name = os.path.join(model_share_dir, model_basename)
            with open(model_name, "wb") as f:
                f.write(program.desc.serialize_to_string())


    def decrypt_model(self, mpc_model_dir, plain_model_path, mpc_model_filename=None, plain_model_filename=None):
        """
        Reveal a paddle model. Load encrypted model (i.e., MPC model shares) from files and decrypt it
        into paddle model.

        Args:
            mpc_model_dir: The directory of all model shares.
            plain_model_path: The directory to save revealed paddle model.
            mpc_model_filename: The name of encrypted model file.
            plain_model_filename: The name of decrypted model file.
        """
        share_dirs = []
        for sub_dir in os.listdir(mpc_model_dir):
            if not sub_dir.startswith("."):
                share_dirs.append(os.path.join(mpc_model_dir, sub_dir))

        place = fluid.CPUPlace()
        exe = fluid.Executor(place=place)
        mpc_model_basename = os.path.basename(
            mpc_model_filename) if mpc_model_filename is not None else MPC_MODEL_NAME

        [main_prog, _, _] = fluid.io.load_inference_model(
            dirname=share_dirs[0], executor=exe, model_filename=mpc_model_basename)
        if main_prog.num_blocks > 1:
            raise NotImplementedError(
                "The number of blocks in current main program"
                "is {}, which is not supported in this version"
                .format(main_prog.num_blocks()))

        global_block = main_prog.global_block()
        g_scope = fluid.global_scope()

        # a set storing unique variables to decrypt
        vars_set = set()

        # store variable name in vars_set, and decrypt dtype and shape
        for mpc_var_name in global_block.vars:
            mpc_var = global_block.var(mpc_var_name)
            if mpc_var.name != "feed" and mpc_var.name != "fetch":
                vars_set.add(mpc_var.name)
                if mpc_var_name in plain_vars:
                    # var.desc.set_dtype(fluid.framework.convert_np_dtype_to_dtype_(np.float64))
                    continue
                else:
                    plain_var_shape = mpc_var.shape
                    mpc_var.desc.set_shape(plain_var_shape)
                    mpc_var.desc.set_dtype(fluid.framework.convert_np_dtype_to_dtype_(np.float32))

        # remove init op
        first_mpc_op = global_block.ops[0]
        if first_mpc_op.type == 'mpc_init':
            global_block._remove_op(0)

        # decrypt op type, or other attrs if needed
        for mpc_op in global_block.ops:
            # rename ops
            if str(mpc_op.type).startswith(MPC_OP_PREFIX):
                new_type = str(mpc_op.type)[len(MPC_OP_PREFIX):]
                mpc_op.desc.set_type(new_type)
            elif mpc_op.type == 'fill_constant':
                mpc_op._set_attr(name='shape', val=(1))
                mpc_op._set_attr(name='value', val=1.0)
                mpc_op._set_attr(name='dtype', val=5)

        # decrypt tensor values for each variable in vars_set
        for var_name in vars_set:
            var = global_block.var(var_name)
            if g_scope.find_var(var_name) is not None:
                param = g_scope.find_var(var_name)
                if var_name in plain_vars:
                    pass
                else:
                    # reconstruct plaintext
                    param_tensor_shares = self._get_param_all_shares(
                        var_name, share_dirs, mpc_model_basename)
                    param_tensor = reconstruct(
                        param_tensor_shares, type=np.float32)
                    param.get_tensor()._set_dims(var.shape)
                    param.get_tensor().set(param_tensor, place)

                fluid.io.save_vars(
                    executor=exe,
                    dirname=plain_model_path,
                    vars=[var],
                    filename=var_name)

        # trigger sync to replace old ops
        op_num = global_block.desc.op_size()
        _ = global_block.desc.append_op()
        global_block.desc._remove_op(op_num, op_num + 1)

        # save plaintext model file.
        model_basename = os.path.basename(
            plain_model_filename) if plain_model_filename is not None else MODEL_NAME
        if not os.path.exists(plain_model_path):
            os.makedirs(plain_model_path)
        model_name = os.path.join(plain_model_path, model_basename)
        with open(model_name, "wb") as f:
            f.write(main_prog.desc.serialize_to_string())


    def _get_param_all_shares(self, param_name, share_dirs, model_file):
        """
        Get all shares of one parameter from directories.

        Args:
            param_name: The name of parameter.
            share_dirs: The directories which storing model shares.
            model_file: The name of model file.

        Returns:
            ndarray. The loaded shares.
        """
        exe = fluid.Executor(place=fluid.CPUPlace())
        param_shares = []
        for share_dir in share_dirs:
            _ = fluid.io.load_inference_model(
                dirname=share_dir, executor=exe, model_filename=model_file)
            g_scope = fluid.global_scope()
            param = g_scope.find_var(param_name)
            param_tensor = np.array(param.get_tensor())
            param_shares.append(param_tensor)
        return np.array(param_shares, dtype=np.int64)


    def _is_supported_op(self, op_name):
        """
        Check if op is supported for encryption and decryption.

        Args:
            op_name: The name of op.

        Returns:
            True if supported.
        """
        if op_name not in supported_mpc_ops:
            if str(op_name).endswith('_grad'):
                self._is_supported_op(str(op_name)[:-5])
            else:
                return False
        return True


    def load_mpc_model(self, exe, mpc_model_dir, mpc_model_filename, inference=False):
        """
        Load MPC model from files. The loaded program of the model would be inserted
        init OP and then switched to default_main_program for further MPC procedure.

        Args:
            exe: The executor used for loading.
            mpc_model_dir: The directory of MPC model.
            mpc_model_filename: The filename of MPC model.
            inference: Whether the model to load is used for inference. If true, the
            model to load should be an inference model, and feed_name, fetch_targets
            would be returned with the loaded program after inserting init OP. Otherwise,
            after inserting init OP, the loaded program would be switched to
            default_main_program and returned. Default value is False.

        Returns:
            default_main_program if inference is False. Otherwise, default_main_program,
            feed_name, and fetch_targets would be returned.
        """
        mpc_program, feed_names, fetch_targets = fluid.io.load_inference_model(executor=exe,
                                      dirname=mpc_model_dir,
                                      model_filename=mpc_model_filename)
        # find init OP
        global_block = fluid.default_main_program().global_block()
        init_op_idx = self._find_init_op_idx(global_block)
        if init_op_idx < 0:
            raise RuntimeError('No mpc_init op in global block, '
                               'maybe you should use paddle_fl.mpc.init() first.')
        init_op = global_block.ops[init_op_idx]
        # find the last feed OP for inserting init OP
        last_feed_op_idx = self._find_last_feed_op_idx(mpc_program.global_block())
        # insert init OP as the first OP of MPC program if no feed OP,
        # otherwise, insert it after the last feed OP.
        insert_idx = 0 if last_feed_op_idx < 0 else last_feed_op_idx + 1
        loaded_mpc_program = self._insert_init_op(main_prog=mpc_program,
                                             init_op=init_op,
                                             index=insert_idx)
        if inference:
            return loaded_mpc_program, feed_names, fetch_targets
        else:
            # switch loaded_mpc_program to default_main_program
            fluid.framework.switch_main_program(loaded_mpc_program)
            return fluid.default_main_program()


    def _find_init_op_idx(self, block):
        """
        Find the index of mpc_init op.

        Args:
            block: The block of program.

        Returns:
            The index of mpc_init op.
        """
        for idx, op in enumerate(block.ops):
            if op.type == 'mpc_init':
                return idx
        return -1


    def _find_last_feed_op_idx(self, block):
        """
        Find the index of the last feed OP.

        Args:
            block: The block of program.

        Returns:
            The index of the last feed OP.
        """
        feed_idx = -1
        for idx, op in enumerate(block.ops):
            if op.type == 'feed':
                feed_idx = idx
        return feed_idx


    def save_trainable_model(self, exe, model_dir, model_filename=None, program=None):
        """
        Save trainable model, which includes saving program and
        persistable parameters into files. The saved model can be
        loaded by fluid.io.load_inference_model for further training
        or updating.

        Args:
            exe: The executor used for saving.
            model_dir: The directory of model to save.
            model_filename: The filename of model to save.
            program: The program to save, default to default_main_program.

        TODO: can move this to paddle_mpc/python/paddle_fl/mpc/io.py
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_basename = os.path.basename(
            model_filename) if model_filename is not None else MPC_MODEL_NAME
        # save program
        model_name = os.path.join(model_dir, model_basename)
        if program is None:
            program = fluid.default_main_program()
        with open(model_name, "wb") as f:
            f.write(program.desc.serialize_to_string())
        # save parameters
        fluid.io.save_persistables(executor=exe,
                                   dirname=model_dir,
                                   main_program=program)


    def _insert_init_op(self, main_prog, init_op, index):
        """
        Insert init OP into main_prog according to the index.

        Args:
            main_prog: The program to insert init OP.
            init_op: The init OP for MPC running.
            index: The place that the init_op to insert.

        Returns:
            The program after inserting init OP.
        """
        main_prog.global_block()._sync_with_cpp()
        op_desc = main_prog.global_block().desc._insert_op(index)
        mpc_init_op = fluid.framework.Operator(block=main_prog.global_block(),
                                               desc=op_desc,
                                               type=init_op.type,
                                               attrs=init_op.all_attrs())
        main_prog.global_block().ops.insert(index, mpc_init_op)
        return main_prog


class Aby3DataUtils(DataUtils):
    """
    Aby3DataUtils
    """

    def __init__(self):
        """
        init
        """
        self.SHARE_NUM = 3
        self.PRE_SHAPE = (2, )
        self.MPC_ONE_SHARE = mdu.aby3_one_share

    def encrypt(self, number):
        """
        Encrypts the plaintext number into three secret shares
        Args:
            number: float, the number to share
        Returns:
            three shares of input number
        """
        try:
            return mdu.aby3_share(number)
        except Exception as e:
            raise RuntimeError(e.message)


    def decrypt(self, shares):
        """
        Reveal plaintext value from raw secret shares
        Args:
            shares: shares to reveal from (list)
        Return:
           the plaintext number (float)
        """
        try:
            return mdu.aby3_reveal(shares)
        except Exception as e:
            raise RuntimeError(e.message)

    def get_shares(self, shares, index):
        """
        Build ABY3 shares from raw shares according to index
        Args:
            shares: the input raw share array, expected to have shape of [3, ...]
            index: index of the ABY3 share, should be 0, 1, or 2
        Returns:
            ABY3 shares array corresponding to index, e.g.:
                [shares[index % 3], shares[(index + 1) %3]]
        Examples:
            input_shares: [3, 2, 4], where 3 is the dim of raw shares
            index: 0
            output: [input_shares[0], input_shares[1]], shape = (2, 2, 4)
        """
        if index < 0 or index >= self.SHARE_NUM:
            raise ValueError("Index should fall in (0..2) but now: {}".format(
                index))

        if shares.size % self.SHARE_NUM != 0 or shares.shape[0] != self.SHARE_NUM:
            raise ValueError("Shares to split has incorrect shape: {}".format(
                shares.shape))

        first = index % self.SHARE_NUM
        second = (index + 1) % self.SHARE_NUM
        return np.array([shares[first], shares[second]], dtype=np.int64)


    def save_shares(self, share_reader, part_name):
        """
        Combine raw shares to ABY3 shares, and persists to files. Each ABY3 share will be
        put into the corresponding file, e.g., ${part_name}.part[0..2]. For example,
            [share0, share1] -> ${part_name}.part0
            [share1, share2] -> ${part_name}.part1
            [share2, share0] -> ${part_name}.part2

        Args:
            share_reader: iteratable function object returning a single array of raw shares
                in shape of [3, ...] each time
            part_name: file name
        Returns:
            files with names of ${part_name}.part[0..2]
         """
        exts = [".part0", ".part1", ".part2"]
        with open(part_name + exts[0], 'wb') as file0, \
            open(part_name + exts[1], 'wb') as file1, \
            open(part_name + exts[2], 'wb') as file2:

            files = [file0, file1, file2]
            for shares in share_reader():
                for idx in six.moves.range(0, 3): # 3 parts
                    share = self.get_shares(shares, idx)
                    files[idx].write(share.tostring())

    def reconstruct(self, aby3_shares, type=np.float):
        """
        Reconstruct plaintext from ABY3 shares

        Args:
            aby3_shares: all the three ABY3 share arrays, each is of shape (2, dims), where the share slices
                are stored rowwise
            type: expected type of final result
        Returns:
            plaintext array reconstructed from the three ABY3 shares, with shape of (dims)
        Example:
            aby3_shares: three ABY3 shares of shape [2, 2]
                shares[0]: [[a0, b0], [a1, b1]]
                shares[1]: [[a1, b1], [a2, b2]]
                shares[2]: [[a2, b2], [a0, b0]]
            output:
                [a, b], where a = decrypt(a0, a1, a2), b = decrypt(b0, b1, b2)
        """
        if len(aby3_shares) != self.SHARE_NUM: # should collect shares from 3 parts
            raise ValueError("Number of aby3 shares should be 3 but was: {}".
                             format(len(aby3_shares)))

        raw_shares = aby3_shares[:, 0]
        data_shape = raw_shares.shape[1:] # get rid of the first dim of [3, xxx]
        data_size = np.prod(data_shape)
        row_first_raw_shares = raw_shares.reshape(self.SHARE_NUM, data_size).transpose(1, 0)

        result = np.empty((data_size, ), dtype=type)
        for idx in six.moves.range(0, data_size):
            result[idx] = self.decrypt(row_first_raw_shares[idx].tolist())

        return result.reshape(data_shape)


    def batch(self, reader, batch_size, drop_last=False):
        """
        A batch reader return a batch data meeting the shared data's shape.
        E.g., a batch arrays with shape (2, 3, 4) of batch_size will be transform to (2, batch_size, 3, 4),
        where the first dim 2 is the number of secret shares in ABY3.

        Args: see paddle.batch method
        Returns: the batched reader
        """
        paddle_batch_reader = paddle.batch(reader, batch_size, drop_last)

        def reshaped_batch_reader():
            """
            internal reader
            """
            r = paddle_batch_reader()
            for instance in r:
                perm = np.arange(0, len(np.array(instance).shape), 1)
                # permute the first two axes
                perm[0], perm[1] = perm[1], perm[0]
                yield np.transpose(instance, perm)

        return reshaped_batch_reader



class PrivcDataUtils(DataUtils):
    """
    PrivcDataUtils
    """

    def __init__(self):
        """
        init
        """
        self.SHARE_NUM = 2
        self.PRE_SHAPE = ()
        self.MPC_ONE_SHARE = mdu.privc_one_share


    def encrypt(self, number):
        """
        Encrypts the plaintext number into two secret shares
        Args:
            number: float, the number to share
        Returns:
            two shares of input number
        """
        try:
            return mdu.privc_share(number)
        except Exception as e:
            raise RuntimeError(e.message)


    def decrypt(self, shares):
        """
        Reveal plaintext value from raw secret shares
        Args:
            shares: shares to reveal from (list)
        Return:
           the plaintext number (float)
        """
        try:
            return mdu.privc_reveal(shares)
        except Exception as e:
            raise RuntimeError(e.message)


    def get_shares(self, shares, index):
        """
        Build share from raw shares according to index

        Args:
            shares: the input raw share array, expected to have shape of [SHARE_NUM, ...]
            index: index of the privc share, should be 0 or 1
        Returns:
            share corresponding to index, e.g.: shares[index]
        """
        if index < 0 or index >= self.SHARE_NUM:
            raise ValueError("Index should fall in {0, {}} but now: {}".format(
                self.SHARE_NUM, index))

        if shares.size % self.SHARE_NUM != 0 or shares.shape[0] != self.SHARE_NUM:
            raise ValueError("Shares to split has incorrect shape: {}".format(
                shares.shape))

        return np.array(shares[index], dtype=np.int64)


    def save_shares(self, share_reader, part_name):
        """
        Combine raw shares to privc shares, and persists to files. Each privc share will be
        put into the corresponding file, e.g., ${part_name}.part[0..1]. For example,
            share0 -> ${part_name}.part0
            share1 -> ${part_name}.part1

        Args:
            share_reader: iteratable function object returning a single array of raw shares
                in shape of [2, ...] each time
            part_name: file name
        Returns:
            files with names of ${part_name}.part[0..1]
        """
        exts = [".part0", ".part1"]
        with open(part_name + exts[0], 'wb') as file0, \
            open(part_name + exts[1], 'wb') as file1:

            files = [file0, file1]
            for shares in share_reader():
                for idx in six.moves.range(0, 2):
                    share = self.get_shares(shares, idx)
                    files[idx].write(share.tostring())

    def reconstruct(self, privc_shares, type=np.float):
        """
        Reconstruct plaintext from privc shares

        Args:
            privc_shares: all the two privc share arrays, where the share slices
                          are stored rowwise
            type: expected type of final result
        Returns:
            plaintext array reconstructed from the two privc shares, with shape of (dims)
        Example:
            privc_shares: two privc shares of shape [2]
                shares[0]: [a0, b0]
                shares[1]: [a1, b1]
            output:
                [a, b], where a = decrypt(a0, a1), b = decrypt(b0, b1)
        """
        if len(privc_shares) != self.SHARE_NUM: # should collect shares from 2 parts
            raise ValueError("Number of privc shares should be 2 but was: {}".
                             format(len(privc_shares)))

        raw_shares = privc_shares
        data_shape = raw_shares.shape[1:] # get rid of the first dim of [2, xxx]
        data_size = np.prod(data_shape)
        row_first_raw_shares = raw_shares.reshape(self.SHARE_NUM, data_size).transpose(1, 0)

        result = np.empty((data_size, ), dtype=type)
        for idx in six.moves.range(0, data_size):
            result[idx] = self.decrypt(row_first_raw_shares[idx].tolist())

        return result.reshape(data_shape)


    def batch(self, reader, batch_size, drop_last=False):
        """
        A batch reader return a batch data meeting the shared data's shape.
        E.g., a batch arrays with shape (3, 4) of batch_size will be transform to (batch_size, 3, 4).

        Args: see paddle.batch method
        Returns: the batched reader
        """
        paddle_batch_reader = paddle.batch(reader, batch_size, drop_last)

        def reshaped_batch_reader():
            """
            internal reader
            """
            r = paddle_batch_reader()
            for instance in r:
                yield np.array(instance)

        return reshaped_batch_reader


data_utils_list =  {
    'aby3': Aby3DataUtils(),
    'privc': PrivcDataUtils()}

def get_datautils(protocol_name):
    return data_utils_list[protocol_name]
