# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid

# paddle.enable_static()

class OpWarpper(object):
    def __init__(self, op):
        self.op = op
        self.input_names = OpWarpper.get_input_var_names(self.op)
        self.output_names = OpWarpper.get_output_var_names(self.op)
        self.owner = None
        for name in self.input_names:
            ''' var name: <host name>|<slot name> '''
            split_str = "|"
            if split_str not in name:
                continue
            self.owner = name.split(split_str)[0]
            if self.owner not in ["Host", "Customer"]:
                raise ValueError(
                        "Failed to split: owner_name({}) can only ".format(name) +\
                        "be defined as 'Host' or 'Customer' at present.")
    
    @staticmethod
    def get_input_var_names(op):
        names = []
        for name in op.input_names:
            names.extend(op.input(name))
        return set(names)

    @staticmethod
    def get_output_var_names(op):
        names = []
        for name in op.output_names:
            names.extend(op.output(name))
        return set(names)

    def is_predecessor(self, op):
        for name in op.input_names:
            if name in self.output_names:
                return True
        return False

    def is_successor(self, op):
        for name in op.output_names:
            if name in self.input_names:
                return True
        return False

    def is_optimize_op(self):
        return self.op._is_optimize_op()
    
    def is_backward_op(self):
        return self.op._is_backward_op()

    def __str__(self):
        return "[{}] type: {}, inputs: {}, outputs: {}".format(
                self.owner, self.op.type, self.input_names, self.output_names)


class ProgramWarpper(object):
    def __init__(self, program):
        self.program = program.clone()
        self.vars = {var.name: var for var in self.program.list_vars()}
        self.ops = [OpWarpper(op) for op in self.program.global_block().ops]
        op_relationship = self._build_op_relationship(self.ops)
        self.forward_edges = op_relationship[0]
        self.backward_egdes = op_relationship[1]
    
    def _build_op_relationship(self, ops):
        forward_edges = {op: [] for op in ops} # op -> list of op
        backward_egdes = {op: [] for op in ops}
        for idx, op in enumerate(ops):
            for peer_idx in range(idx + 1, len(ops)):
                peer_op = ops[peer_idx]
                if op.is_predecessor(peer_op):
                    forward_edges[op].append(peer_op)
                    backward_egdes[peer_op].append(op)
        return [forward_edges, backward_egdes]

    def fliter_ops(self, ftyps, prefix):
        ret = []
        for op in self.ops:
            if self._multi_fliter(op, ftyps, prefix):
                ret.append(op)
        return ret

    def _multi_fliter(self, op, ftypes, prefix):
        for ftype in ftypes:
            if not self._fliter(op, ftype, prefix):
                return False
        return True

    def _fliter(self, op, ftype, prefix):
        table = {
            "forward": self.__filter_forward,
            "not_forward": self.__filter_not_forward,
            "input_contains_prefix": self.__filter_input_contains_prefix,
            "output_contains_prefix": self.__filter_output_contains_prefix,
            "input_has_grad": self.__filter_input_has_grad,
            "input_has_no_grad": self.__filter_input_has_no_grad,
            "output_has_grad": self.__filter_output_has_grad,
            "output_has_no_grad": self.__filter_output_has_no_grad,
            "input_contains_prefix_without_grad": self.__filter_input_contains_prefix_without_grad,
            "output_contains_prefix_without_grad": self.__filter_output_contains_prefix_without_grad,
            "input_contains_prefix_with_grad": self.__filter_input_contains_prefix_with_grad,
            "output_contains_prefix_with_grad": self.__filter_output_contains_prefix_with_grad,
        }
        return table[ftype](op, prefix)
        
    def __filter_forward(self, op, prefix):
        if op.is_optimize_op() or op.is_backward_op():
            return False
        return True

    def __filter_not_forward(self, op, prefix):
        return not self.__filter_forward(op, prefix)

    def __filter_input_contains_prefix(self, op, prefix):
        for name in op.input_names:
            if name.startswith(prefix):
                return True
        return False
    
    def __filter_output_contains_prefix(self, op, prefix):
        for name in op.output_names:
            if name.startswith(prefix):
                return True
        return False

    def __filter_input_contains_prefix_without_grad(self, op, prefix):
        for name in op.input_names:
            if name.startswith(prefix) and not name.endswith("@GRAD"):
                return True
        return False

    def __filter_output_contains_prefix_without_grad(self, op, prefix):
        for name in op.output_names:
            if name.startswith(prefix) and not name.endswith("@GRAD"):
                return True
        return False

    def __filter_input_contains_prefix_with_grad(self, op, prefix):
        for name in op.input_names:
            if name.startswith(prefix) and name.endswith("@GRAD"):
                return True
        return False

    def __filter_output_contains_prefix_with_grad(self, op, prefix):
        for name in op.output_names:
            if name.startswith(prefix) and name.endswith("@GRAD"):
                return True
        return False

    def __filter_input_has_grad(self, op, prefix):
        for name in op.input_names:
            if name.endswith("@GRAD"):
                return True
        return False

    def __filter_input_has_no_grad(self, op, prefix):
        return not self.__filter_input_has_grad(op, prefix)

    def __filter_output_has_grad(self, op, prefix):
        for name in op.output_names:
            if name.endswith("@GRAD"):
                return True
        return False

    def __filter_output_has_no_grad(self, op, prefix):
        return not self.__filter_output_has_grad(op, prefix)


class Reformer(object):
    def __init__(self):
        pass

    @staticmethod
    def split_program_by_name(program):
        prefix = "Host|"
        return Reformer.split_program_by_key_prefix(program, prefix)
    
    @staticmethod
    def split_program_by_key_prefix(program, prefix):
        #p1_program = Reformer.clone_head_sub_program_by_key_prefix(program, prefix)
        p1_program = Reformer.remain_head_sub_program_by_key_prefix(program, prefix)
        p2_program = Reformer.remain_middle_sub_program_by_key_prefix(program, prefix)
        p3_program = Reformer.remain_tail_sub_program_by_key_prefix(program, prefix)
        return [p1_program, p2_program, p3_program]

    @staticmethod
    def clone_head_sub_program_by_key_prefix(src_program, key_prefix):
        program_warp = ProgramWarpper(src_program)
        # 以有后的op作为分界op（包含在dst中）
        fliter_list = ["forward", "output_contains_prefix_without_grad"]
        demarcations = program_warp.fliter_ops(fliter_list, key_prefix)
        
        predecessor_ops = set()
        # demarcations need be clone
        for op in demarcations:
            if op not in predecessor_ops:
                Reformer._get_predecessors(program_warp, op, predecessor_ops)
                predecessor_ops.add(op)
        ops = set([op_warp.op for op_warp in predecessor_ops])
        return Reformer._clone_sub_program_by_given_ops(program_warp, ops)
    
    @staticmethod
    def remain_head_sub_program_by_key_prefix(src_program, key_prefix):
        program_warp = ProgramWarpper(src_program)
        # 以有前的op做为分界op（包含在dst中）
        fliter_list = ["forward", "output_contains_prefix_without_grad"]
        demarcations = program_warp.fliter_ops(fliter_list, key_prefix)
        
        predecessor_ops = set()
        # demarcations need be clone
        for op in demarcations:
            if op not in predecessor_ops:
                Reformer._get_predecessors(program_warp, op, predecessor_ops)
                predecessor_ops.add(op)
        ops = set([op_warp.op for op_warp in predecessor_ops])
        return Reformer._remain_sub_program_by_given_ops(program_warp, ops)

    @staticmethod
    def remain_tail_sub_program_by_key_prefix(src_program, key_prefix):
        program_warp = ProgramWarpper(src_program)
        # 以有前的op做为分界op（包含在dst中）
        fliter_list = ["not_forward", "input_contains_prefix_with_grad"]
        demarcations = program_warp.fliter_ops(fliter_list, key_prefix)
        
        successor_ops = set()
        # demarcations need be remain
        for op in demarcations:
            if op not in successor_ops:
                Reformer._get_successors(program_warp, op, successor_ops)
                successor_ops.add(op)
        ops = set([op_warp.op for op_warp in successor_ops])
        return Reformer._remain_sub_program_by_given_ops(program_warp, ops)

    @staticmethod
    def remain_middle_sub_program_by_key_prefix(src_program, key_prefix):
        program_warp = ProgramWarpper(src_program)
        fliter_list = ["forward", "output_contains_prefix_without_grad"]
        head_demarcations = program_warp.fliter_ops(fliter_list, key_prefix)
        predecessor_ops = set()
        # head_demarcations need be cut
        for op in head_demarcations:
            if op not in predecessor_ops:
                Reformer._get_predecessors(program_warp, op, predecessor_ops)
                predecessor_ops.add(op)
        
        fliter_list = ["not_forward", "input_contains_prefix_with_grad"]
        tail_demarcations = program_warp.fliter_ops(fliter_list, key_prefix)
        successor_ops = set()
        # tail_demarcations need be cut
        for op in tail_demarcations:
            if op not in successor_ops:
                Reformer._get_successors(program_warp, op, successor_ops)
                successor_ops.add(op)

        remove_ops = predecessor_ops | successor_ops
        ops = set([op_warp.op for op_warp in remove_ops])
        return Reformer._remove_sub_program_by_given_ops(program_warp, ops)

    @staticmethod
    def _get_predecessors(program_warp, op, predecessors):
        for peer in program_warp.backward_egdes[op]:
            if peer not in predecessors:
                predecessors.add(peer)
                Reformer._get_predecessors(program_warp, peer, predecessors)

    @staticmethod
    def _get_successors(program_warp, op, successors):
        for peer in program_warp.forward_edges[op]:
            if peer not in successors:
                successors.add(peer)
                Reformer._get_successors(program_warp, peer, successors)

    @staticmethod
    def _clone_sub_program_by_given_ops(program_warp, ops):
        def add_var_depended_by_op(dst_program, op):
            input_names =  OpWarpper.get_input_var_names(op)
            output_names =  OpWarpper.get_output_var_names(op)
            required = input_names | output_names
            block = dst_program.global_block()
            for name in required:
                if not block.has_var(name):
                    var = program_warp.vars[name]
                    block._clone_variable(var, force_persistable=var.persistable)
        
        def add_op(dst_program, op):
            add_var_depended_by_op(dst_program, op)
            inputs = {name: op.input(name) for name in op.input_names}
            outputs = {name: op.output(name) for name in op.output_names}
            attrs = {name: op.attr(name) for name in op.attr_names}
            block = dst_program.global_block()
            block.append_op(
                    type=op.type,
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs)
        
        dst_program = fluid.framework.Program()
        for op in program_warp.program.global_block().ops:
            if op in ops:
                add_op(dst_program, op)
        return dst_program

    @staticmethod
    def _remove_sub_program_by_given_ops(program_warp, ops):
        def remove_op(dst_program, op):
            block = dst_program.global_block()
            block._remove_op(op.idx)
        
        def clear_vars(dst_program):
            block = dst_program.global_block()
            used_var_names = set()
            for op in block.ops:
                input_names =  OpWarpper.get_input_var_names(op)
                output_names =  OpWarpper.get_output_var_names(op)
                used_var_names.update(input_names)
                used_var_names.update(output_names)
            for var_name in program_warp.vars:
                if var_name not in used_var_names:
                    block._remove_var(var_name)

        dst_program = program_warp.program
        for op in ops:
            remove_op(dst_program, op)
        clear_vars(dst_program)
        return dst_program        

    @staticmethod
    def _remain_sub_program_by_given_ops(program_warp, ops):
        removed_ops = set()
        for op in program_warp.program.global_block().ops:
            if op not in ops:
                removed_ops.add(op)
        return Reformer._remove_sub_program_by_given_ops(program_warp, removed_ops)
