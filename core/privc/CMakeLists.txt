set(PRIVC_SRCS
    "privc_context.cc"
    "integer.cc"
    "bit.cc"
    "ot.cc"
    "common_utils.cc"
    #"../paddlefl_mpc/mpc_protocol/context_holder.cc"
)

add_library(privc_o OBJECT ${PRIVC_SRCS})
add_dependencies(privc_o crypto common mpc_tensor)

add_library(privc STATIC $<TARGET_OBJECTS:privc_o>)

target_link_libraries(privc common mpc_tensor)

cc_test(privc_fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc)
cc_test(triplet_generator_test SRCS triplet_generator_test.cc DEPS privc)
#cc_test(privc_gc_test SRCS privc_gc_test.cc DEPS privc)
cc_test(privc_bit_test SRCS privc_bit_test.cc DEPS privc)
cc_test(privc_integer_test SRCS privc_integer_test.cc DEPS privc)
cc_test(privc_fixedpoint_test SRCS privc_fixedpoint_test.cc DEPS privc)

