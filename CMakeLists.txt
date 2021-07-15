cmake_minimum_required(VERSION 3.15)

project(PaddleEncrypted)

add_compile_options(-msse4.2 -fPIC -DPADDLE_WITH_MKLDNN -O2  -Wno-ignored-attributes)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if (WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIX dll)
elseif (APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIX dylib)
    set(CMAKE_FIND_LIBRARY_PREFIXES lib)
elseif (LINUX)
    set(CMAKE_FIND_LIBRARY_SUFFIX so)
    set(CMAKE_FIND_LIBRARY_PREFIXES lib)
endif()

if (NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python)
endif()

find_program(PYTHON ${PYTHON_EXECUTABLE})
if (NOT PYTHON)
  message(FATAL_ERROR "${PYTHON_EXECUTABLE} not found")
endif()

option(WITH_TESTING "Compile with unit testing" ON)

option(WITH_PSI "Compile with psi lib" ON)

option(WITH_GRPC "Compile with grpc lib" ON)

option(USE_AES_NI "Compile with AES NI" ON)

option(USE_OPENMP "Compile with OpenMP" ON)

option(USE_ABY3_TRUNC1 "Compile with ABY3 truncate 1 algorithm" OFF)

option(BUILD_PADDLE_FROM_SOURCE "build paddle from source" OFF)

########################### the project build part ###############################

include(third_party)
include(generic)

include_directories(.)

if (USE_AES_NI)
    add_compile_definitions(USE_AES_NI)
    add_compile_options(-maes)
endif (USE_AES_NI)

if (USE_OPENMP)
    add_compile_options(-fopenmp)
    find_package(OpenMP REQUIRED)
endif(USE_OPENMP)

if (USE_ABY3_TRUNC1)
    add_compile_definitions(USE_ABY3_TRUNC1)
endif(USE_ABY3_TRUNC1)

add_subdirectory(core/common)
add_subdirectory(core/privc)
add_subdirectory(core/privc3)
add_subdirectory(core/paddlefl_mpc/mpc_protocol)
add_subdirectory(core/paddlefl_mpc/operators)
add_subdirectory(core/paddlefl_mpc/data_utils)

if (WITH_TESTING)
    add_subdirectory(core/testing)
endif()

if (WITH_PSI)
    add_subdirectory(core/psi)
endif()

add_library(fluid_framework ALIAS paddle_framework)

# generate dynamic .so lib
add_library(paddle_enc SHARED
    $<TARGET_OBJECTS:common_o>
    $<TARGET_OBJECTS:mpc_tensor_o>
    $<TARGET_OBJECTS:mpc_protocol_o>
    $<TARGET_OBJECTS:mpc_ops_o>
    $<TARGET_OBJECTS:privc_o>)

target_link_libraries(paddle_enc "-Wl,--whole-archive ${ZLIB_LIBRARIES} -Wl,--no-whole-archive" zlib)
target_link_libraries(paddle_enc gmp)
target_link_libraries(paddle_enc gmpxx)
target_link_libraries(paddle_enc seal)
target_link_libraries(paddle_enc gloo)
if (WITH_GRPC)
    add_definitions(-DWITH_GRPC)
    target_link_libraries(paddle_enc grpc++)
    target_link_libraries(paddle_enc grpc++_unsecure)
    target_link_libraries(paddle_enc grpc_unsecure)
    target_link_libraries(paddle_enc gpr)
endif()
target_link_libraries(paddle_enc hiredis)
target_link_libraries(paddle_enc crypto)
target_link_libraries(paddle_enc fluid_framework)

set(CMAKE_SKIP_INSTALL_RPATH TRUE)

set(PADDLE_ENCRYPTED_LIB_PATH "${CMAKE_SOURCE_DIR}/python/paddle_fl/mpc/libs")
if (WITH_GRPC)
    install(DIRECTORY "${THIRD_PARTY_PATH}/install/grpc/lib/"
        DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party)
endif()
install(DIRECTORY "${THIRD_PARTY_PATH}/install/gloo/lib/"
    DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party)
install(DIRECTORY "${THIRD_PARTY_PATH}/install/hiredis/lib/"
    DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party)
install(DIRECTORY "${THIRD_PARTY_PATH}/install/openssl/lib/"
    DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party)

install(TARGETS paddle_enc mpc_data_utils
    LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}
    LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})

if (WITH_PSI)
    install(TARGETS psi LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
endif()
