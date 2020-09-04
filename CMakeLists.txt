cmake_minimum_required(VERSION 3.15)

project(PaddleEncrypted)

add_compile_options(-msse4.2 -fPIC -DPADDLE_WITH_MKLDNN -O2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)

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

execute_process(COMMAND ${PYTHON} -c "import paddle;print(paddle.version.full_version)"
  RESULT_VARIABLE ret OUTPUT_VARIABLE paddle_version OUTPUT_STRIP_TRAILING_WHITESPACE)

if (NOT ret)
  if (NOT ${paddle_version} STRGREATER_EQUAL "1.8.0")
    message(FATAL_ERROR "Paddle installation of >= 1.8.0 is required but ${paddle_version} is found")
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

option(WITH_TESTING "Compile with unit testing" ON)

option(WITH_PSI "Compile with psi lib" ON)

option(USE_AES_NI "Compile with AES NI" ON)

option(USE_OPENMP "Compile with OpenMP" ON)

option(USE_ABY3_TRUNC1 "Compile with ABY3 truncate 1 algorithm" OFF)

########################### the project build part ###############################
message(STATUS "Using paddlepaddle installation of ${paddle_version}")
message(STATUS "paddlepaddle include directory: ${PADDLE_INCLUDE}")
message(STATUS "paddlepaddle libraries directory: ${PADDLE_LIB}")
message(STATUS "python libraries directory: ${PYTHON_SITE_PACKAGES}")

include(third_party)
include(generic)

include_directories(.)
include_directories(${PADDLE_INCLUDE})
include_directories(${PADDLE_INCLUDE}/third_party)

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

add_library(fluid_framework SHARED IMPORTED GLOBAL)
set_property(TARGET fluid_framework PROPERTY IMPORTED_LOCATION ${FLUID_LIB})

# generate dynamic .so lib
add_library(paddle_enc SHARED
    $<TARGET_OBJECTS:privc3_o>
    $<TARGET_OBJECTS:mpc_protocol_o>
    $<TARGET_OBJECTS:mpc_ops_o>)

target_link_libraries(paddle_enc fluid_framework)
target_link_libraries(paddle_enc gloo)
target_link_libraries(paddle_enc hiredis)

set(CMAKE_SKIP_INSTALL_RPATH TRUE)

set(PADDLE_ENCRYPTED_LIB_PATH "${CMAKE_SOURCE_DIR}/python/paddle_fl/mpc/libs")

install(DIRECTORY "${THIRD_PARTY_PATH}/install/gloo/lib/"
    DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party)
install(DIRECTORY "${THIRD_PARTY_PATH}/install/hiredis/lib/"
    DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party)
install(DIRECTORY "${THIRD_PARTY_PATH}/install/openssl/lib/"
    DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}/third_party/openssl)

install(TARGETS paddle_enc mpc_data_utils
    LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH}
    LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})

if (WITH_PSI)
    install(TARGETS psi LIBRARY DESTINATION ${PADDLE_ENCRYPTED_LIB_PATH})
endif()
