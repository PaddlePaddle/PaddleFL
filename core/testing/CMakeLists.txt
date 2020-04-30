# for paddle mpc test case
if(WITH_TESTING)
    # TODO: add gflas to accept cmd args?
    # TODO: add other deps libs
    cc_library(paddle_mpc_gtest_main SRCS paddle_mpc_gtest_main.cc DEPS mpc_protocol gtest)
endif()
