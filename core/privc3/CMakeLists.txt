
add_library(privc3 INTERFACE)

target_link_libraries(privc3 INTERFACE common mpc_tensor)

cc_test(fixedpoint_util_test SRCS fixedpoint_util_test.cc DEPS privc3)

cc_test(boolean_tensor_test SRCS boolean_tensor_test.cc DEPS privc3)

cc_test(fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc3)
