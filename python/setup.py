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
Setup file for python code install
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from setuptools import find_packages
from setuptools import setup
from version import fl_version

def python_version():
    """
    get python version
    """
    return [int(v) for v in platform.python_version().split(".")]


max_version, mid_version, min_version = python_version()

REQUIRED_PACKAGES = [
    'six >= 1.10.0', 'protobuf >= 3.1.0', 'paddlepaddle >= 1.8.0', 'paddlepaddle-gpu >= 1.8'
]

if max_version < 3:
    REQUIRED_PACKAGES += ["enum"]
else:
    REQUIRED_PACKAGES += ["numpy"]

REQUIRED_PACKAGES += ["unittest2"]
packages = ['paddle_fl.paddle_fl', 'paddle_fl.paddle_fl.common', 'paddle_fl.paddle_fl.core',
            'paddle_fl.paddle_fl.dataset', 'paddle_fl.paddle_fl.reader', 'paddle_fl.mpc',
            'paddle_fl.mpc.layers', 'paddle_fl.mpc.data_utils','paddle_fl', 'paddle_fl.paddle_fl.common',
            'paddle_fl.paddle_fl.dataset', 'paddle_fl.paddle_fl.reader', 'paddle_fl.paddle_fl.core',
             'paddle_fl.paddle_fl.core.master', 'paddle_fl.paddle_fl.core.scheduler',
             'paddle_fl.paddle_fl.core.server', 'paddle_fl.paddle_fl.core.submitter',
             'paddle_fl.paddle_fl.core.trainer', 'paddle_fl.paddle_fl.core.trainer.diffiehellman',
              'paddle_fl.paddle_fl.core.strategy', 'paddle_fl.paddle_fl.core.strategy.details']
package_data = {
    'paddle_fl.mpc': [
        'libs/*', 'libs/third_party/*', 'libs/third_party/openssl/*',
        'libs/third_party/openssl/engines/*'
    ]
}
package_dir = {
    'paddle_fl.paddle_fl': './paddle_fl/paddle_fl',
    'paddle_fl.paddle_fl.common': './paddle_fl/paddle_fl/common',
    'paddle_fl.paddle_fl.core': './paddle_fl/paddle_fl/core',
    'paddle_fl.paddle_fl.dataset': './paddle_fl/paddle_fl/dataset',
    'paddle_fl.paddle_fl.reader': './paddle_fl/paddle_fl/reader',
    'paddle_fl.mpc': './paddle_fl/mpc',
    'paddle_fl.mpc.layers': './paddle_fl/mpc/layers',
    'paddle_fl.mpc.data_utils': './paddle_fl/mpc/data_utils',
    'paddle_fl': './paddle_fl',
    'paddle_fl.paddle_fl.common': './paddle_fl/paddle_fl/common',
    'paddle_fl.paddle_fl.dataset': './paddle_fl/paddle_fl/dataset',
    'paddle_fl.paddle_fl.reader': './paddle_fl/paddle_fl/reader',
    'paddle_fl.paddle_fl.core': './paddle_fl/paddle_fl/core',
    'paddle_fl.paddle_fl.core.master': './paddle_fl/paddle_fl/core/master',
    'paddle_fl.paddle_fl.core.scheduler': './paddle_fl/paddle_fl/core/scheduler',
    'paddle_fl.paddle_fl.core.server': './paddle_fl/paddle_fl/core/server',
    'paddle_fl.paddle_fl.core.submitter': './paddle_fl/paddle_fl/core/submitter',
    'paddle_fl.paddle_fl.core.trainer': './paddle_fl/paddle_fl/core/trainer',
    'paddle_fl.paddle_fl.core.trainer.diffiehellman': './paddle_fl/paddle_fl/core/trainer/diffiehellman',
    'paddle_fl.paddle_fl.core.strategy': './paddle_fl/paddle_fl/core/strategy',
    'paddle_fl.paddle_fl.core.strategy.details': './paddle_fl/paddle_fl/core/strategy/details',
}


setup(
    name='paddle_fl',
    version=fl_version.replace('-', ''),
    description=(
        'Privacy-Preserving Deep Learning Package Based on PaddlePaddle.'),
    long_description='',
    url='https://github.com/PaddlePaddle/PaddleFL',
    author='PaddlePaddle Author',
    author_email='paddle-dev@baidu.com',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    package_data=package_data,
    package_dir=package_dir,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=(
        'paddle_fl paddlepaddle multi-task transfer distributed-training multiparty-computation privacy-preserving'
    ))
