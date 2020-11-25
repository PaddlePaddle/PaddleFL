set(PRIVC_SRCS
    "privc_context.cc"
)

add_library(privc_o OBJECT ${PRIVC_SRCS})
add_dependencies(privc_o crypto common mpc_tensor)

add_library(privc SHARED $<TARGET_OBJECTS:privc_o>)

target_link_libraries(privc common mpc_tensor)

cc_test(privc_fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc)

