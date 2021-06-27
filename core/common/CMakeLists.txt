set(COMMON_SRCS
    "./aes.cc"
    "./naorpinkas_ot.cc"
    "./prng.cc"
    "./rand_utils.cc"
    "./sse_transpose.cc"
)


add_library(common_o OBJECT ${COMMON_SRCS})
add_dependencies(common_o crypto)

add_library(common STATIC $<TARGET_OBJECTS:common_o>)

set(common_deps_libs crypto dnnl iomp5)

if (USE_OPENMP)
    list(APPEND common_deps_libs OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif (USE_OPENMP)

target_link_libraries(common ${common_deps_libs})

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

set(mpc_tensor_deps_libs fluid_framework common)

target_link_libraries(mpc_tensor ${mpc_tensor_deps_libs})

cc_test(paddle_tensor_test SRCS paddle_tensor_test.cc DEPS mpc_tensor)
