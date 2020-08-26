set(PSI_SRCS
    "./aes.cc"
    "./cuckoo_hash.cc"
    "./naorpinkas_ot.cc"
    "./prng.cc"
    "./psi.cc"
    "./psi_api.cc"
    "./rand_utils.cc"
    "./sse_transpose.cc"
)

add_library(psi_o OBJECT ${PSI_SRCS})
add_dependencies(psi_o crypto)

add_library(psi SHARED $<TARGET_OBJECTS:psi_o>)

if (USE_OPENMP)
    target_link_libraries(psi OpenMP::OpenMP_CXX OpenMP::OpenMP_C crypto)
else()
    target_link_libraries(psi crypto)
endif (USE_OPENMP)

cc_test(aes_test SRCS aes_test.cc DEPS psi)
cc_test(ot_test SRCS ot_test.cc DEPS psi)
cc_test(psi_test SRCS psi_test.cc DEPS psi)
cc_test(psi_api_test SRCS psi_api_test.cc DEPS psi)
