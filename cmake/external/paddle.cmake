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

if(BUILD_PADDLE_FROM_SOURCE)

SET(PADDLE_PROJECT       "extern_paddle")
IF((NOT DEFINED PADDLE_VER) OR (NOT DEFINED PADDLE_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(PADDLE_TAG "v1.8.5" CACHE STRING "" FORCE)
  SET(PADDLE_NAME "paddle" CACHE STRING "" FORCE)
  SET(PADDLE_URL "https://github.com/PaddlePaddle/Paddle.git" CACHE STRING "" FORCE)
ENDIF()

MESSAGE(STATUS "PADDLE_NAME: ${PADDLE_NAME}, PADDLE_URL: ${PADDLE_URL}")

SET(PADDLE_PREFIX_DIR    "${THIRD_PARTY_PATH}/paddle")
SET(PADDLE_SOURCE_DIR   "${THIRD_PARTY_PATH}/paddle/src/extern_paddle")
SET(PADDLE_BUILD_DIR   "${THIRD_PARTY_PATH}/paddle/src/extern_paddle-build")
SET(PADDLE_INSTALL_DIR   "${THIRD_PARTY_PATH}/install/paddle")

cache_third_party(${PADDLE_PROJECT}
    REPOSITORY   ${PADDLE_URL}
    TAG          ${PADDLE_TAG}
    DIR          PADDLE_SOURCE_DIR)

ExternalProject_Add(
    ${PADDLE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${PADDLE_DOWNLOAD_CMD}"
    PREFIX                ${PADDLE_PREFIX_DIR}
    SOURCE_DIR            ${PADDLE_SOURCE_DIR}
    BINARY_DIR            ${PADDLE_BUILD_DIR}
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
                    -DCMAKE_INSTALL_PREFIX=${PADDLE_INSTALL_ROOT}
                    -DWITH_GPU=${WITH_GPU}
                    -DWITH_MKL=ON
                    -DWITH_DISTRIBUTE=ON
                    -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${PADDLE_INSTALL_DIR}
)

include_directories(${PADDLE_SOURCE_DIR})
include_directories(${PADDLE_BUILD_DIR})
include_directories(${PADDLE_BUILD_DIR}/third_party/install/glog/include)
include_directories(${PADDLE_BUILD_DIR}/third_party/install/gflags/include)
include_directories(${PADDLE_BUILD_DIR}/third_party/boost/src/extern_boost)
include_directories(${PADDLE_BUILD_DIR}/third_party/install/mkldnn/include/)
include_directories(${PADDLE_BUILD_DIR}/third_party/eigen3/src/extern_eigen3/)
include_directories(${PADDLE_BUILD_DIR}/third_party/dlpack/src/extern_dlpack/include)
include_directories(${PADDLE_BUILD_DIR}/third_party/install/xxhash/include/)
include_directories(${PADDLE_BUILD_DIR}/third_party/threadpool/src/extern_threadpool)

ADD_LIBRARY(paddle_framework SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET paddle_framework PROPERTY IMPORTED_LOCATION ${PADDLE_BUILD_DIR}/python/paddle/libs/libpaddle_framework.so)
ADD_DEPENDENCIES(paddle_framework extern_paddle)

ADD_LIBRARY(dnnl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET dnnl PROPERTY IMPORTED_LOCATION ${PADDLE_BUILD_DIR}/third_party/install/mkldnn/lib/libdnnl.so)

ADD_LIBRARY(iomp5 SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET iomp5 PROPERTY IMPORTED_LOCATION ${PADDLE_BUILD_DIR}/third_party/install/mklml/lib/libiomp5.so)

else()

execute_process(COMMAND ${PYTHON} -c "import paddle;print(paddle.version.full_version)"
  RESULT_VARIABLE ret OUTPUT_VARIABLE paddle_version OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT ret)
  if (NOT ${paddle_version} STRGREATER_EQUAL "1.8.5")
    message(FATAL_ERROR "Paddle installation of >= 1.8.5 is required but ${paddle_version} is found")
  endif()
else()
  message(FATAL_ERROR "Could not get paddle version.")
endif()
execute_process(COMMAND ${PYTHON} -c "import paddle; print(paddle.sysconfig.get_include())"
    OUTPUT_VARIABLE PADDLE_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON} -c "import paddle; print(paddle.sysconfig.get_lib())"
    OUTPUT_VARIABLE PADDLE_LIB OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

find_library(FLUID_LIB NAMES paddle_framework PATHS ${PADDLE_LIB})

if (NOT FLUID_LIB)
    message(FATAL_ERROR "paddle_framework library is not found in ${PADDLE_LIB}")
endif()

message(STATUS "Using paddlepaddle installation of ${paddle_version}")
message(STATUS "paddlepaddle include directory: ${PADDLE_INCLUDE}")
message(STATUS "paddlepaddle libraries directory: ${PADDLE_LIB}")
message(STATUS "python libraries directory: ${PYTHON_SITE_PACKAGES}")

include_directories(${PADDLE_INCLUDE})
include_directories(${PADDLE_INCLUDE}/third_party)

add_library(paddle_framework SHARED IMPORTED GLOBAL)
set_property(TARGET paddle_framework PROPERTY IMPORTED_LOCATION ${FLUID_LIB})

ADD_LIBRARY(dnnl SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET dnnl PROPERTY IMPORTED_LOCATION ${PADDLE_LIB}/libdnnl.so)

ADD_LIBRARY(iomp5 SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET iomp5 PROPERTY IMPORTED_LOCATION ${PADDLE_LIB}/libiomp5.so)

endif()
