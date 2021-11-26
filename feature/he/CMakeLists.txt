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
