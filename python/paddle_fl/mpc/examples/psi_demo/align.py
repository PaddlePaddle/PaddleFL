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
Data alignment.
"""
import argparse
import paddle_fl.mpc.data_utils.alignment as alignment


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--party_id", type=int, help="the id of this party")
    parser.add_argument("--endpoints", type=str,
                        default='0:127.0.0.1:11111,1:127.0.0.1:22222',
                        help="id:ip:port info")
    parser.add_argument("--data_file", type=str, help="data file")
    parser.add_argument("--is_receiver", action='store_true', help="whether is receiver")
    args = parser.parse_args()
    return args


def do_align(args):
    """
    Do alignment.
    """
    # read data from file
    input_set = set()
    for line in open(args.data_file, 'r'):
        input_set.add(line.strip())
    # do alignment
    result = alignment.align(input_set=input_set,
                             party_id=args.party_id,
                             endpoints=args.endpoints,
                             is_receiver=args.is_receiver)
    return result


if __name__ == '__main__':
    # use signal for interrupt from keyboard
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    args = parse_args()
    print('ARGUMENTS: party_id={}, endpoints={}, is_receiver={}, data_file={}'
          .format(args.party_id, args.endpoints, args.is_receiver, args.data_file))
    align_rst = do_align(args)
    print("Alignment result is: {}".format(align_rst))
