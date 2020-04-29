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

INCLUDE(ExternalProject)

SET(HIREDIS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(HIREDIS_INCLUDE_DIRS ${HIREDIS_DIR})
SET(HIREDIS_NAME "hiredis" CACHE STRING "" FORCE)
SET(HIREDIS_URL "https://paddlefl.bj.bcebos.com/hiredis.tar.gz" CACHE STRING "" FORCE)
ExternalProject_Add(
   extern_hiredis
   #URL https://paddlefl.bj.bcebos.com/hiredis.tar.gz
#   GIT_REPOSITORY "https://github.com/redis/hiredis"
   PREFIX ${CMAKE_CURRENT_BINARY_DIR}/hiredis
   SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/hiredis
   DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/hiredis
   DOWNLOAD_COMMAND wget --no-check-certificate ${HIREDIS_URL} -c -q -O ${HIREDIS_NAME}.tar.gz
                       && tar xf ${HIREDIS_NAME}.tar.gz --strip-components 1
   BUILD_IN_SOURCE 1
   CONFIGURE_COMMAND ""
   BUILD_COMMAND make static
   INSTALL_COMMAND "")
INCLUDE_DIRECTORIES(${HIREDIS_INCLUDE_DIRS})
message(STATUS "${HIREDIS_INCLUDE_DIRS}")
ADD_LIBRARY(hiredis SHARED IMPORTED GLOBAL)
ADD_DEPENDENCIES(hiredis extern_hiredis)