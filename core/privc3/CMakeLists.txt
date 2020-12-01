set(PRIVC3_SRCS
    "./aes.cc"
    "./paddle_tensor.cc"
    "./prng.cc"
    "./rand_utils.cc"
    "./tensor_adapter_factory.cc"
)

if (USE_AES_NI)
    add_compile_definitions(USE_AES_NI)
endif (USE_AES_NI)

add_library(privc3_o OBJECT ${PRIVC3_SRCS})

if (WITH_GPU)
    cuda_add_library(privc3_cu SHARED paddle_tensor_impl.cu)
endif (WITH_GPU)

add_dependencies(privc3_o fluid_framework)

add_library(privc3 STATIC $<TARGET_OBJECTS:privc3_o>)

set(privc3_deps_libs fluid_framework crypto)

if (USE_OPENMP)
    list(APPEND privc3_deps_libs OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif (USE_OPENMP)

if (WITH_GPU)
    list(APPEND privc3_deps_libs privc3_cu)
endif (WITH_GPU)

target_link_libraries(privc3 ${privc3_deps_libs})

cc_test(fixedpoint_util_test SRCS fixedpoint_util_test.cc DEPS privc3)
cc_test(paddle_tensor_test SRCS paddle_tensor_test.cc DEPS privc3)
cc_test(boolean_tensor_test SRCS boolean_tensor_test.cc DEPS privc3)

cc_test(fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc3)
