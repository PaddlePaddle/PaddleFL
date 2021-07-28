import paddle.fluid as fluid
import numpy as np
import yaml
import logging
import json

from paddle_fl.split_learning import HostExecutor
from paddle_fl.split_learning.core.table import LocalTable
from paddle_fl.split_learning.core.reader import TmpReader
import network

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=logging.DEBUG)

_LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    data_path = sys.argv[1]
    host_input, consumer_input, label, prediction, cost = network.net()
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    place = fluid.CPUPlace()
    table = LocalTable(data_path)
    reader = TmpReader(place)
    exe = HostExecutor(place, table, reader)
    exe.load_program_from_full_network(
            startup_program=startup_program,
            main_program=main_program)
    exe.start(7858)
