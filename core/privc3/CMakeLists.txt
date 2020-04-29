add_compile_options(-msse4.2 -maes)

set(PRIVC3_SRCS
    "./aes.cc"
    "./paddle_tensor.cc"
    "./prng.cc"
    "./rand_utils.cc"
    "./tensor_adapter_factory.cc"
)

add_library(privc3_o OBJECT ${PRIVC3_SRCS})
add_dependencies(privc3_o fluid_framework)

add_library(privc3 STATIC $<TARGET_OBJECTS:privc3_o>)
target_link_libraries(privc3 fluid_framework)

cc_test(fixedpoint_util_test SRCS fixedpoint_util_test.cc DEPS privc3)
cc_test(paddle_tensor_test SRCS paddle_tensor_test.cc DEPS privc3)
cc_test(boolean_tensor_test SRCS boolean_tensor_test.cc DEPS privc3)

cc_test(fixedpoint_tensor_test SRCS fixedpoint_tensor_test.cc DEPS privc3)

#set(CMAKE_BUILD_TYPE "Debug")
