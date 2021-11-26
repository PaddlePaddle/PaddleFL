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
if (WITH_GRPC)
    add_definitions(-DWITH_GRPC)
endif()

option(USE_AES_NI "Compile with AES NI" ON)

option(USE_OPENMP "Compile with OpenMP" ON)

option(USE_ABY3_TRUNC1 "Compile with ABY3 truncate 1 algorithm" OFF)

option(BUILD_PADDLE_FROM_SOURCE "build paddle from source" OFF)

option(WITH_GPU "Compile with gpu" ON)

########################### the project build part ###############################

if (WITH_GPU)
    find_package(CUDA)
    add_definitions(-DUSE_CUDA)
    add_definitions(-DPADDLE_WITH_MKLDNN)
    add_definitions(-DPADDLE_WITH_CUDA)
    add_definitions(-DEIGEN_USE_GPU)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-std=c++14")
    list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
    list(APPEND CUDA_NVCC_FLAGS "-w")
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CUDA_NVCC_FLAGS "-O3 -DNDEBUG")

    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})
endif(WITH_GPU)

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
add_subdirectory(feature/he)
if (NOT WITH_GPU)
    add_subdirectory(core/privc)
endif()
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

if (WITH_GRPC)
    set(PROTO_SRCS "./core/paddlefl_mpc/mpc_protocol/network/mesh_network_grpc.cc")
    set(GRPC_DEPS grpc++_unsecure grpc_unsecure gpr zlib protobuf)
    grpc_library(transport_o SRCS ${PROTO_SRCS} PROTO ./core/paddlefl_mpc/mpc_protocol/network/protos/transport.proto DEPS ${GRPC_DEPS})
endif()

# generate dynamic .so lib
if (WITH_GPU)
add_library(paddle_enc SHARED
    $<TARGET_OBJECTS:common_o>
    $<TARGET_OBJECTS:mpc_tensor_o>
    $<TARGET_OBJECTS:mpc_protocol_o>
    $<TARGET_OBJECTS:mpc_ops_o>)
else()
add_library(paddle_enc SHARED
    $<TARGET_OBJECTS:common_o>
    $<TARGET_OBJECTS:mpc_tensor_o>
    $<TARGET_OBJECTS:mpc_protocol_o>
    $<TARGET_OBJECTS:mpc_ops_o>
    $<TARGET_OBJECTS:transport_o>
    $<TARGET_OBJECTS:privc_o>)
endif()

target_link_libraries(paddle_enc "-Wl,--whole-archive ${ZLIB_LIBRARIES} -Wl,--no-whole-archive" zlib)
target_link_libraries(paddle_enc gmp)
target_link_libraries(paddle_enc gmpxx)
target_link_libraries(paddle_enc seal)
target_link_libraries(paddle_enc gloo)
if (WITH_GRPC)
    target_link_libraries(paddle_enc grpc++)
    target_link_libraries(paddle_enc grpc++_unsecure)
    target_link_libraries(paddle_enc grpc_unsecure)
    target_link_libraries(paddle_enc gpr)
    target_link_libraries(paddle_enc transport_o)
endif()
target_link_libraries(paddle_enc hiredis)
target_link_libraries(paddle_enc crypto)
target_link_libraries(paddle_enc fluid_framework)

if (WITH_GPU)
    target_link_libraries(paddle_enc mpc_protocol_cu)
    target_link_libraries(paddle_enc mpc_tensor_cu)
    target_link_libraries(paddle_enc common_cu)
    target_link_libraries(paddle_enc "-Wl,-no-as-needed" mpc_ops_cu)
endif (WITH_GPU)

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

set(FEATURE_LIB "${CMAKE_SOURCE_DIR}/feature/python/libs")
install(TARGETS he_utils LIBRARY DESTINATION ${FEATURE_LIB})
 
if (WITH_PSI)
    install(TARGETS psi LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
endif()

if (WITH_GPU)
    install(TARGETS mpc_tensor_cu LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
    install(TARGETS mpc_protocol_cu LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
    install(TARGETS common_cu LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
    install(TARGETS mpc_ops_cu LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
endif (WITH_GPU)
