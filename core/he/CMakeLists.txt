set(PYBIND_HE_SRCS
    "./he_utils.cc"
    "./paillier.cc"
)

if (NOT PYTHON_INCLUDE_DIRS)
  find_package(PythonLibs REQUIRED)
endif()

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(he_utils MODULE ${PYBIND_HE_SRCS})
target_link_libraries(he_utils PRIVATE pybind gmp gmpxx)
set_target_properties(he_utils PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")

set(FE_PROTO_MODULE_PATH "paddle_fl/feature_engineering/proto")
py_proto_grpc_compile(my_target PROTO "${CMAKE_SOURCE_DIR}/python/${FE_PROTO_MODULE_PATH}/metrics.proto"
                                PROTO_PATH "${CMAKE_SOURCE_DIR}/python")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${FE_PROTO_MODULE_PATH}/metrics_pb2.py"
              "${CMAKE_CURRENT_BINARY_DIR}/${FE_PROTO_MODULE_PATH}/metrics_pb2_grpc.py"
        DESTINATION "${CMAKE_SOURCE_DIR}/python/${FE_PROTO_MODULE_PATH}")

set(FEATURE_LIB "${CMAKE_SOURCE_DIR}/python/paddle_fl/feature_engineering/libs")
install(TARGETS he_utils LIBRARY DESTINATION ${FEATURE_LIB})
