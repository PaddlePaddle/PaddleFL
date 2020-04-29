add_compile_options(-msse4.2 -maes)
 
set(PYBIND_SRCS
    "./data_utils.cc"
)
 
if (NOT PYTHON_INCLUDE_DIRS)
  find_package(PythonLibs REQUIRED)
endif()

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(mpc_data_utils MODULE ${PYBIND_SRCS})
target_link_libraries(mpc_data_utils PRIVATE pybind)
target_link_libraries(mpc_data_utils PRIVATE privc3)
target_link_libraries(mpc_data_utils PRIVATE psi)
set_target_properties(mpc_data_utils PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
