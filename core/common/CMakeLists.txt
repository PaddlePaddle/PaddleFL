set(COMMON_SRCS
    "./aes.cc"
    "./naorpinkas_ot.cc"
    "./prng.cc"
    "./rand_utils.cc"
    "./sse_transpose.cc"
)


add_library(common_o OBJECT ${COMMON_SRCS})
add_dependencies(common_o crypto)

add_library(common SHARED $<TARGET_OBJECTS:common_o>)

if (USE_OPENMP)
    target_link_libraries(common OpenMP::OpenMP_CXX OpenMP::OpenMP_C crypto)
else()
    target_link_libraries(common crypto)
endif (USE_OPENMP)

cc_test(aes_test SRCS aes_test.cc DEPS common)
cc_test(ot_test SRCS ot_test.cc DEPS common)
cc_test(crypto_test SRCS crypto_test.cc DEPS privc)

set(TENSOR_SRCS
    "./paddle_tensor.cc"
    "./tensor_adapter_factory.cc"
)

add_library(mpc_tensor_o OBJECT ${TENSOR_SRCS})
add_dependencies(mpc_tensor_o fluid_framework common)

add_library(mpc_tensor STATIC $<TARGET_OBJECTS:mpc_tensor_o>)

target_link_libraries(mpc_tensor fluid_framework common)

cc_test(paddle_tensor_test SRCS paddle_tensor_test.cc DEPS mpc_tensor)
