set(PRIVC_SRCS
    "privc_context.cc"
)

add_library(privc_o OBJECT ${PRIVC_SRCS})
add_dependencies(privc_o crypto privc3)

add_library(privc SHARED $<TARGET_OBJECTS:privc_o>)

target_link_libraries(privc psi)

#set(CMAKE_BUILD_TYPE Debug)

cc_test(crypto_test SRCS crypto_test.cc DEPS privc)
cc_test(privc_fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc)

