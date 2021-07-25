set(PROTO_SRCS
    "./aby3_protocol.cc"
    "./network/mesh_network.cc"
    "./network/network_factory.cc"
    "./mpc_config_parameters.cc"
    "./context_holder.cc"
    "./mpc_instance.cc"
    "./mpc_protocol_factory.cc"
    "./aby3_operators_impl/elementwise_op.cc"
    "./aby3_operators_impl/common.cc"
    "./aby3_operators_impl/matrix_op.cc"
    "./privc_protocol.cc"
    "./privc_operators_impl/elementwise_op.cc"
    "./privc_operators_impl/matrix_op.cc"
)

add_library(mpc_protocol_o OBJECT ${PROTO_SRCS})
if (WITH_GRPC)
    add_dependencies(mpc_protocol_o gloo hiredis gmp grpc++ fluid_framework)
else()
    add_dependencies(mpc_protocol_o gloo hiredis gmp fluid_framework)
endif()

add_library(mpc_protocol STATIC $<TARGET_OBJECTS:mpc_protocol_o>)
target_link_libraries(mpc_protocol ${GRPC_DEPS} gloo hiredis privc3 privc_o fluid_framework)

if (WITH_GRPC)
    target_link_libraries(mpc_protocol transport_o)
    cc_test(mesh_network_grpc_test SRCS network/mesh_network_grpc_test.cc DEPS mpc_protocol)
endif()
cc_test(mesh_network_test SRCS network/mesh_network_test.cc DEPS mpc_protocol)
cc_test(mpc_protocol_test SRCS mpc_protocol_test.cc DEPS mpc_protocol privc)
cc_test(mpc_instance_test SRCS mpc_instance_test.cc DEPS mpc_protocol privc)


