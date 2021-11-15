set(PSI_SRCS
    "./cuckoo_hash.cc"
    "./psi.cc"
    "./psi_api.cc"
)

add_library(psi_o OBJECT ${PSI_SRCS})
add_dependencies(psi_o crypto common)

if (BUILD_PADDLE_FROM_SOURCE)
    add_dependencies(psi_o extern_paddle)
endif()

add_library(psi SHARED $<TARGET_OBJECTS:psi_o>)

if (USE_OPENMP)
    target_link_libraries(psi OpenMP::OpenMP_CXX OpenMP::OpenMP_C crypto common)
else()
    target_link_libraries(psi crypto common)
endif (USE_OPENMP)

cc_test(psi_test SRCS psi_test.cc DEPS psi)
cc_test(psi_api_test SRCS psi_api_test.cc DEPS psi)
