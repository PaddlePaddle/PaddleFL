# Copyright (c) 2020 Baidu Inc.. All Rights Reserved.
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

SET(SEAL_PREFIX_DIR  ${THIRD_PARTY_PATH}/seal)
SET(SEAL_SOURCE_DIR  ${THIRD_PARTY_PATH}/seal/src/extern_seal)
SET(SEAL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/seal)
SET(SEAL_INCLUDE_DIR "${SEAL_INSTALL_DIR}/include" CACHE PATH "seal include directory." FORCE)
SET(SEAL_REPOSITORY  https://github.com/microsoft/SEAL.git)

SET(SEAL_TAG        v3.6.1)

string(REGEX MATCH "[0-9]\.[0-9]" LIB-VER "${SEAL_TAG}")
INCLUDE_DIRECTORIES(${SEAL_INCLUDE_DIR}/SEAL-${LIB-VER})

IF(WIN32)
  SET(SEAL_LIBRARIES "${SEAL_INSTALL_DIR}/lib/seal-${LIB-VER}.lib" CACHE FILEPATH "seal library." FORCE)
  SET(SEAL_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4530")
ELSE(WIN32)
  SET(SEAL_LIBRARIES "${SEAL_INSTALL_DIR}/lib/libseal-${LIB-VER}.a" CACHE FILEPATH "seal library." FORCE)
  SET(SEAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
ENDIF(WIN32)

cache_third_party(extern_seal
    REPOSITORY   ${SEAL_REPOSITORY}
    TAG          ${SEAL_TAG}
    DIR          SEAL_SOURCE_DIR)

ExternalProject_Add(
    extern_seal
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${SEAL_DOWNLOAD_CMD}"
    PREFIX          ${SEAL_PREFIX_DIR}
    SOURCE_DIR      ${SEAL_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${SEAL_CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_INSTALL_PREFIX=${SEAL_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${SEAL_INSTALL_DIR}/lib
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
                    -DSEAL_USE_CXX17=OFF
                    -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON
                    -DSEAL_USE_MSGSL=OFF
                    -DSEAL_BUILD_DEPS=ON
                    -DSEAL_USE_ZLIB=OFF
                    -DSEAL_USE_ZSTD=OFF
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${SEAL_INSTALL_DIR}
                     -DCMAKE_INSTALL_LIBDIR:PATH=${SEAL_INSTALL_DIR}/lib
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

ADD_LIBRARY(seal STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET seal PROPERTY IMPORTED_LOCATION ${SEAL_LIBRARIES})
ADD_DEPENDENCIES(seal extern_seal)
