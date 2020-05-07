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

include(ExternalProject)

set(OPENSSL_SOURCES_DIR ${THIRD_PARTY_PATH}/openssl)
set(OPENSSL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openssl)
set(OPENSSL_INCLUDE_DIR "${OPENSSL_INSTALL_DIR}/include")
set(OPENSSL_NAME "openssl")

include(ProcessorCount)
#ProcessorCount(NUM_OF_PROCESSOR)

if((NOT DEFINED OPENSSL_URL) OR (NOT DEFINED OPENSSL_VER))
  message(STATUS "use pre defined download url")
  set(OPENSSL_URL "https://paddlefl.bj.bcebos.com/openssl-1.0.2u.tar.gz" CACHE STRING "" FORCE)
  set(OPENSSL_VER "openssl-1.0.2u" CACHE STRING "" FORCE)
endif()

ExternalProject_Add(
  extern_openssl
  PREFIX            ${OPENSSL_SOURCES_DIR}
  DOWNLOAD_COMMAND  wget --no-check-certificate ${OPENSSL_URL} -c -q -O ${OPENSSL_NAME}.tar.gz
                    && tar -xvf ${OPENSSL_NAME}.tar.gz
  SOURCE_DIR        ${OPENSSL_SOURCES_DIR}/src/${OPENSSL_VER}
  CONFIGURE_COMMAND ./config shared --openssldir=${OPENSSL_INSTALL_DIR}  -lrt -Wl,--no-as-needed
  BUILD_COMMAND     make depend -j ${NUM_OF_PROCESSOR} &&
                    make build_libcrypto -j ${NUM_OF_PROCESSOR} &&
                    make build_apps -j ${NUM_OF_PROCESSOR}
  INSTALL_COMMAND   make install_sw
  BUILD_IN_SOURCE   1
)

set(OPENSSL_CRYPTO_LIBRARY "${OPENSSL_INSTALL_DIR}/lib/libcrypto.so")

add_library(crypto SHARED IMPORTED GLOBAL)
set_property(TARGET crypto PROPERTY IMPORTED_LOCATION ${OPENSSL_CRYPTO_LIBRARY})

add_dependencies(crypto extern_openssl)

include_directories(${OPENSSL_INCLUDE_DIR})
