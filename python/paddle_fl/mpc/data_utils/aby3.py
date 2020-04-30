#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module provide data utilities for ABY3 protocol, including
data encryption, decryption, share save and loading.
"""
import os
import numpy as np
import six
import paddle
import paddle.fluid as fluid
import mpc_data_utils as mdu

__all__ = [
    'encrypt',
    'decrypt',
    'make_shares',
    'get_aby3_shares',
    'save_aby3_shares',
    'load_aby3_shares',
    'reconstruct',
    'batch',
    'encrypt_model',
    'decrypt_model',
]

SHARE_NUM = 3
ABY3_SHARE_DIM = 2
ABY3_MODEL_NAME = "__model__.aby3"
MODEL_NAME = "__model__"
MODEL_SHARE_DIR = "model_share"
MPC_OP_PREFIX = "mpc_"


def encrypt(number):
    """
    Encrypts the plaintext number into three secret shares
    Args:
        number: float, the number to share
    Returns:
        three shares of input number
    """
    try:
        return mdu.share(number)
    except Exception as e:
        raise RuntimeError(e.message)


def decrypt(shares):
    """
    Reveal plaintext value from raw secret shares
    Args:
        shares: shares to reveal from (list)
    Return:
       the plaintext number (float)
    """
    try:
        return mdu.reveal(shares)
    except Exception as e:
        raise RuntimeError(e.message)


def make_shares(num_array):
    """
    Create raw shares for an array.
    
    Args:
        num_array: the input data array
    Returns:
        3 shares of the num_array in type of ndarray
    Example:
        input array with shape [2, 2]: [[a, b], [c, d]]
        output shares with shape [3, 2, 2]:
        [[[a0, b0],
          [c0, d0]],
         [[a1, b1],
          [c1, d1]],
         [[a2, b2],
          [c2, d2]]]
    """
    old_size = num_array.size
    flat_num_array = num_array.reshape(old_size,)
    new_shape = (SHARE_NUM, ) + num_array.shape
    result = np.empty((old_size, SHARE_NUM), dtype=np.int64)
    for idx in six.moves.range(0, old_size):
        result[idx] = encrypt(flat_num_array[idx])

    result = result.transpose(1, 0)
    result = result.reshape(new_shape)
    return result


def get_aby3_shares(shares, index):
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
        output: [input_shares[0], input_shares[1]], shape = (2, 3, 4)
    """
    if index < 0 or index >= SHARE_NUM:
        raise ValueError("Index should fall in (0..2) but now: {}".format(
            index))

    if shares.size % SHARE_NUM != 0 or shares.shape[0] != SHARE_NUM:
        raise ValueError("Shares to split has incorrect shape: {}".format(
            shares.shape))

    first = index % SHARE_NUM
    second = (index + 1) % SHARE_NUM
    return np.array([shares[first], shares[second]], dtype=np.int64)


def save_aby3_shares(share_reader, part_name):
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
                share = get_aby3_shares(shares, idx)
                files[idx].write(share.tostring())


def load_aby3_shares(part_name, id, shape, append_share_dim=True):
    """
    Load ABY3 shares from file with name ${part_name}.part{id} in shape of ${shape}.

    Args:
        part_name and id: use to build the file name of ${part_name}.part{id}
        shape: the shape of output array
        append_share_dim: if true, a dim of 2 will be add to first of shape, which
            means two share slices are in row store
    Returns:
        iteratable function object returing a share array with give shape each time
    """
    if id not in (0, 1, 2):
        raise ValueError("Illegal id: {}, should 0, 1 or 2".format(id))

    if append_share_dim == True:
        shape = (ABY3_SHARE_DIM, ) + shape
    elif shape[0] != ABY3_SHARE_DIM or len(
            shape) < 2:  # an aby3 share has at least 2 dimensions
        raise ValueError("Illegal ABY3 share shape: {}".format(shape))

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


def reconstruct(aby3_shares, type=np.float):
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
    if len(aby3_shares) != 3: # should collect shares from 3 parts
        raise ValueError("Number of aby3 shares should be 3 but was: {}".
                         format(len(aby3_shares)))

    raw_shares = aby3_shares[:, 0]
    data_shape = raw_shares.shape[1:] # get rid of the first dim of [3, xxx]
    data_size = np.prod(data_shape)
    row_first_raw_shares = raw_shares.reshape(3, data_size).transpose(1, 0)

    result = np.empty((data_size, ), dtype=type)
    for idx in six.moves.range(0, data_size):
        result[idx] = decrypt(row_first_raw_shares[idx].tolist())

    return result.reshape(data_shape)


def batch(reader, batch_size, drop_last=False):
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


def encrypt_model(plain_model, mpc_model_dir, model_filename=None):
    """
    Encrypts model, and save to files for mpc inference.

    Args:
        plain_model: The directory of paddle model.
        mpc_model_dir: The directory that save mpc model shares.
        model_filename: The name of model file.
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    [main_prog, _, _] = fluid.io.load_inference_model(
        dirname=plain_model, executor=exe, model_filename=model_filename)
    # TODO(xukun): support more blocks. Tips: may be just adding "for loop" for all blocks.
    if main_prog.num_blocks > 1:
        raise NotImplementedError(
            "The number of blocks in current main program"
            "is {}, which is not supported in this version."
            .format(main_prog.num_blocks()))
    global_block = main_prog.global_block()
    g_scope = fluid.global_scope()
    for op in global_block.ops:
        if op.type != "feed" and op.type != "fetch":
            # TODO: needs to check if the mpc op exists
            op.desc.set_type(MPC_OP_PREFIX + op.type)

        for input_arg_name in op.input_arg_names:
            var = global_block.var(input_arg_name)
            if var.name != "feed" and var.name != "fetch":
                # set mpc param shape = [2, old_shape]
                encrypted_var_shape = (ABY3_SHARE_DIM, ) + var.shape
                var.desc.set_shape(encrypted_var_shape)

                if g_scope.find_var(input_arg_name) is not None:
                    param = g_scope.find_var(input_arg_name)
                    param_tensor_shares = make_shares(
                        np.array(param.get_tensor()))

                    for idx in six.moves.range(SHARE_NUM):
                        param.get_tensor()._set_dims(encrypted_var_shape)
                        param.get_tensor().set(
                            get_aby3_shares(param_tensor_shares, idx), place)

                        param_share_dir = os.path.join(
                            mpc_model_dir, MODEL_SHARE_DIR + "_" + str(idx))
                        fluid.io.save_vars(
                            executor=exe,
                            dirname=param_share_dir,
                            vars=[var],
                            filename=input_arg_name)
    # trigger sync to replace old ops
    op_num = global_block.desc.op_size()
    _ = global_block.desc.append_op()
    global_block.desc._remove_op(op_num, op_num + 1)

    # save mpc model file
    model_basename = os.path.basename(
        model_filename) if model_filename is not None else ABY3_MODEL_NAME
    for idx in six.moves.range(SHARE_NUM):
        model_share_dir = os.path.join(mpc_model_dir,
                                       MODEL_SHARE_DIR + "_" + str(idx))
        if not os.path.exists(model_share_dir):
            os.makedirs(model_share_dir)
        model_name = os.path.join(model_share_dir, model_basename)
        with open(model_name, "wb") as f:
            f.write(main_prog.desc.serialize_to_string())


def decrypt_model(mpc_model_dir, plain_model_path, model_filename=None):
    """
    Reveal a paddle model.

    Args:    
        mpc_model_dir: The directory of all model shares.
        plain_model_path: The directory to save revealed paddle model.
        model_filename: The name of model file.
    """
    share_dirs = []
    for sub_dir in os.listdir(mpc_model_dir):
        if not sub_dir.startswith("."):
            share_dirs.append(os.path.join(mpc_model_dir, sub_dir))

    place = fluid.CPUPlace()
    exe = fluid.Executor(place=place)
    mpc_model_basename = os.path.basename(
        model_filename) if model_filename is not None else ABY3_MODEL_NAME

    [main_prog, _, _] = fluid.io.load_inference_model(
        dirname=share_dirs[0], executor=exe, model_filename=mpc_model_basename)
    if main_prog.num_blocks > 1:
        raise NotImplementedError(
            "The number of blocks in current main program"
            "is {}, which is not supported in this version"
            .format(main_prog.num_blocks()))

    global_block = main_prog.global_block()
    g_scope = fluid.global_scope()
    for op in global_block.ops:
        # rename ops
        if str(op.type).startswith(MPC_OP_PREFIX):
            new_type = str(op.type)[len(MPC_OP_PREFIX):]
            op.desc.set_type(new_type)

        for input_arg_name in op.input_arg_names:
            var = global_block.var(input_arg_name)
            if var.name != "feed" and var.name != "fetch":
                if var.shape[0] != ABY3_SHARE_DIM:
                    raise ValueError(
                        "Variable:{} shape: {} in saved model should start with 2."
                        .format(var.name, var.shape))
                plain_var_shape = var.shape[1:]
                old_var_shape = var.shape
                var.desc.set_shape(plain_var_shape)

                if g_scope.find_var(input_arg_name) is not None:
                    param = g_scope.find_var(input_arg_name)
                    # reconstruct plaintext
                    param_tensor_shares = _get_param_all_shares(
                        input_arg_name, share_dirs, mpc_model_basename)
                    param_tensor = reconstruct(
                        param_tensor_shares, type=np.float32)
                    param.get_tensor()._set_dims(plain_var_shape)
                    param.get_tensor().set(param_tensor, place)

                    fluid.io.save_vars(
                        executor=exe,
                        dirname=plain_model_path,
                        vars=[var],
                        filename=input_arg_name)
    # trigger sync to replace old ops
    op_num = global_block.desc.op_size()
    _ = global_block.desc.append_op()
    global_block.desc._remove_op(op_num, op_num + 1)

    # save plaintext model file.
    model_basename = os.path.basename(
        model_filename) if model_filename is not None else MODEL_NAME
    if not os.path.exists(plain_model_path):
        os.makedirs(plain_model_path)
    model_name = os.path.join(plain_model_path, model_basename)
    with open(model_name, "wb") as f:
        f.write(main_prog.desc.serialize_to_string())


def _get_param_all_shares(param_name, share_dirs, model_file):
    """
    Get all shares of one parameter from directories.

    Args:
        param_name: The name of parameter.
        share_dirs: The directories which storing model shares.
        model_file: The name of model file.
    :return:
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
