# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
include(ExternalProject)

SET(NCCL_PROJECT       "extern_nccl")
IF((NOT DEFINED NCCL_VER) OR (NOT DEFINED NCCL_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(NCCL_VER "a46ea105830fbc3fbb222a00a82862df49cc9a9f" CACHE STRING "" FORCE)
  SET(NCCL_NAME "nccl" CACHE STRING "" FORCE)
  SET(NCCL_URL "https://github.com/NVIDIA/nccl.git" CACHE STRING "" FORCE)
ENDIF()

MESSAGE(STATUS "NCCL_NAME: ${NCCL_NAME}, NCCL_URL: ${NCCL_URL}")

SET(NCCL_SOURCE_DIR   "${THIRD_PARTY_PATH}/nccl/src/extern_nccl")
SET(NCCL_INSTALL_DIR   "${NCCL_SOURCE_DIR}/build")
SET(NCCL_INCLUDE_DIR   "${NCCL_INSTALL_DIR}/include" CACHE PATH "nccl include directory." FORCE)

INCLUDE_DIRECTORIES(${NCCL_INCLUDE_DIR})

cache_third_party(${NCCL_PROJECT}
    REPOSITORY   ${NCCL_URL}
    TAG          ${NCCL_VER}
    DIR          NCCL_SOURCE_DIR)

include(ProcessorCount)
ProcessorCount(NUM_OF_PROCESSOR)

ExternalProject_Add(
    ${NCCL_PROJECT}
    ${SHALLOW_CLONE}
    "${NCCL_DOWNLOAD_CMD}"
    SOURCE_DIR            ${NCCL_SOURCE_DIR}
    UPDATE_COMMAND        ""
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND CXX=${CMAKE_CXX_COMPILER} CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
                  NVCC=${CUDA_NVCC_EXECUTABLE} BUILDDIR=${NCCL_INSTALL_DIR}
                  VERBOSE=0 make -j ${NUM_OF_PROCESSOR}
    INSTALL_COMMAND ""
    )

ADD_LIBRARY(nccl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET nccl PROPERTY IMPORTED_LOCATION ${NCCL_INSTALL_DIR}/lib/libnccl.so)
