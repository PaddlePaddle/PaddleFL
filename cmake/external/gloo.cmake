# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

SET(GLOO_PROJECT       "extern_gloo")
IF((NOT DEFINED GLOO_VER) OR (NOT DEFINED GLOO_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(GLOO_VER "dcfd77389571ec7a3170418f8ed775972e934083" CACHE STRING "" FORCE)
  SET(GLOO_NAME "gloo" CACHE STRING "" FORCE)
  SET(GLOO_URL "https://github.com/facebookincubator/gloo.git" CACHE STRING "" FORCE)
ENDIF()

MESSAGE(STATUS "GLOO_NAME: ${GLOO_NAME}, GLOO_URL: ${GLOO_URL}")

SET(GLOO_PREFIX_DIR    "${THIRD_PARTY_PATH}/gloo")
SET(GLOO_SOURCE_DIR   "${THIRD_PARTY_PATH}/gloo/src/extern_gloo")
SET(GLOO_INSTALL_DIR   "${THIRD_PARTY_PATH}/install/gloo")
SET(GLOO_INCLUDE_DIR   "${GLOO_INSTALL_DIR}/include" CACHE PATH "gloo include directory." FORCE)

INCLUDE_DIRECTORIES(${GLOO_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${HIREDIS_INCLUDE_DIR}/hiredis)

cache_third_party(${GLOO_PROJECT}
    REPOSITORY   ${GLOO_URL}
    TAG          ${GLOO_VER}
    DIR          GLOO_SOURCE_DIR)

ExternalProject_Add(
    ${GLOO_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${GLOO_DOWNLOAD_CMD}"
    DEPENDS         hiredis
    PREFIX                ${GLOO_PREFIX_DIR}
    SOURCE_DIR            ${GLOO_SOURCE_DIR}
    UPDATE_COMMAND        ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    -DCMAKE_INSTALL_PREFIX=${GLOO_INSTALL_ROOT}
                    -DUSE_REDIS=on
                    -DHIREDIS_INCLUDE_DIRS=${HIREDIS_INCLUDE_DIR}/hiredis
                    -DHIREDIS_LIBRARIES=${HIREDIS_INSTALL_DIR}/lib
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${GLOO_INSTALL_DIR}
)

ADD_LIBRARY(gloo STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gloo PROPERTY IMPORTED_LOCATION ${GLOO_INSTALL_DIR}/lib/libgloo.a)
ADD_DEPENDENCIES(gloo extern_gloo gflags)
