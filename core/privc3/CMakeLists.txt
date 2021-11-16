
add_library(privc3 INTERFACE)

target_link_libraries(privc3 INTERFACE common mpc_tensor)

cc_test(fixedpoint_util_test SRCS fixedpoint_util_test.cc DEPS common)

if (WITH_GPU)
    #nv_test(fixedpoint_tensor_test_cu SRCS fixedpoint_tensor_test.cu DEPS privc3 common_cu)
    target_link_libraries(privc3 INTERFACE nccl)
    add_dependencies(privc3 extern_nccl)
else ()
    cc_test(boolean_tensor_test SRCS boolean_tensor_test.cc DEPS privc3)
    cc_test(fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc3)
endif ()
