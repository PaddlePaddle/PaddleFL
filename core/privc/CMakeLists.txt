set(PRIVC_SRCS
    "privc_context.cc"
    "ot.cc"
)

add_library(privc_o OBJECT ${PRIVC_SRCS})
add_dependencies(privc_o crypto common mpc_tensor extern_seal extern_gmp)

add_library(privc STATIC $<TARGET_OBJECTS:privc_o>)

target_link_libraries(privc crypto common mpc_tensor seal gmp gmpxx)

cc_test(privc_fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc)
cc_test(triplet_generator_test SRCS triplet_generator_test.cc DEPS privc)
cc_test(privc_fixedpoint_util_test SRCS fixedpoint_util_test.cc DEPS privc)
cc_test(he_triplet_test SRCS he_triplet_test.cpp DEPS privc)
