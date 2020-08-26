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
add_dependencies(privc3_o fluid_framework)

add_library(privc3 STATIC $<TARGET_OBJECTS:privc3_o>)

if (USE_OPENMP)
    target_link_libraries(privc3 fluid_framework OpenMP::OpenMP_CXX OpenMP::OpenMP_C crypto)
else()
    target_link_libraries(privc3 fluid_framework crypto)
endif (USE_OPENMP)

cc_test(fixedpoint_util_test SRCS fixedpoint_util_test.cc DEPS privc3)
cc_test(paddle_tensor_test SRCS paddle_tensor_test.cc DEPS privc3)
cc_test(boolean_tensor_test SRCS boolean_tensor_test.cc DEPS privc3)

cc_test(fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc3)
