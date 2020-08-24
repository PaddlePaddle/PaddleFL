set(PROTO_SRCS
    "./aby3_protocol.cc"
    "./mesh_network.cc"
    "./mpc_config_parameters.cc"
    "./context_holder.cc"
    "./mpc_instance.cc"
    "./mpc_protocol_factory.cc"
)
add_library(mpc_protocol_o OBJECT ${PROTO_SRCS})
add_dependencies(mpc_protocol_o fluid_framework gloo hiredis)

add_library(mpc_protocol STATIC $<TARGET_OBJECTS:mpc_protocol_o>)
target_link_libraries(mpc_protocol fluid_framework gloo hiredis privc3)

cc_test(mesh_network_test SRCS mesh_network_test.cc DEPS mpc_protocol)
cc_test(mpc_protocol_test SRCS mpc_protocol_test.cc DEPS mpc_protocol)
cc_test(mpc_instance_test SRCS mpc_instance_test.cc DEPS mpc_protocol)


