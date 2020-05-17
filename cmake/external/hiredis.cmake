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

INCLUDE(ExternalProject)

SET(HIREDIS_PREFIX_DIR     ${THIRD_PARTY_PATH}/hiredis)
SET(HIREDIS_SOURCE_DIR     ${THIRD_PARTY_PATH}/hiredis/src/extern_hiredis)
SET(HIREDIS_INSTALL_DIR    ${THIRD_PARTY_PATH}/install/hiredis)
SET(HIREDIS_INCLUDE_DIR    ${HIREDIS_INSTALL_DIR}/include)
SET(HIREDIS_LIBRARY        ${HIREDIS_INSTALL_DIR}/lib/libhiredis.a)
SET(HIREDIS_REPOSITORY     https://github.com/redis/hiredis.git)
SET(HIREDIS_TAG            v0.13.3)

cache_third_party(extern_hiredis
    REPOSITORY    ${HIREDIS_REPOSITORY}
    TAG           ${HIREDIS_TAG}
    DIR           HIREDIS_SOURCE_DIR)

INCLUDE_DIRECTORIES(${HIREDIS_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${HIREDIS_INCLUDE_DIR}/hiredis)

include(ProcessorCount)

ExternalProject_Add(
    extern_hiredis
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${HIREDIS_DOWNLOAD_CMD}"
    DEPENDS           ${HIREDIS_DEPENDS}
    CONFIGURE_COMMAND ""#${HIREDIS_CONFIGURE_COMMAND}
    PREFIX            ${HIREDIS_PREFIX_DIR}
    SOURCE_DIR        ${HIREDIS_SOURCE_DIR}
    #UPDATE_COMMAND    ""
    BUILD_COMMAND     CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
                      CFLAGS=${CMAKE_C_FLAGS} DEBUG=${CMAKE_C_FLAGS_DEBUG}
                      make -j ${NUM_OF_PROCESSOR}
    INSTALL_COMMAND   PREFIX=${HIREDIS_INSTALL_DIR} INCLUDE_PATH=include/hiredis
                      LIBRARY_PATH=lib make install
    BUILD_IN_SOURCE   1
)

ADD_LIBRARY(hiredis SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET hiredis PROPERTY IMPORTED_LOCATION ${HIREDIS_LIBRARY})
ADD_DEPENDENCIES(hiredis extern_hiredis)
