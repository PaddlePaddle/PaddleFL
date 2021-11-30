set(PROTO_SRCS
    "./aby3_protocol.cc"
    "./network/mesh_network.cc"
    "./network/network_factory.cc"
    "./mpc_config_parameters.cc"
    "./context_holder.cc"
    "./mpc_instance.cc"
    "./mpc_protocol_factory.cc"
)

if (NOT WITH_GPU)
    list(APPEND PROTO_SRCS
        "./aby3_operators_impl/elementwise_op.cc"
        "./aby3_operators_impl/common.cc"
        "./aby3_operators_impl/matrix_op.cc"
        "./privc_protocol.cc"
        "./privc_operators_impl/elementwise_op.cc"
        "./privc_operators_impl/matrix_op.cc"
        "./abstract_context.cc"
        "./aby3_operators.cc"
        )
endif()

if (WITH_GPU)
    set(PROTO_SRCS_CU
        "./aby3_operators_impl/elementwise_op.cu"
        "./aby3_operators_impl/common.cu"
        "./aby3_operators_impl/matrix_op.cu"
        "./abstract_context.cu"
        "./aby3_operators.cu"
        "./context_holder.cc"
        )
endif()

add_library(mpc_protocol_o OBJECT ${PROTO_SRCS})

if (WITH_GRPC)
    add_dependencies(mpc_protocol_o gloo hiredis gmp grpc++ fluid_framework)
else()
    add_dependencies(mpc_protocol_o gloo hiredis gmp fluid_framework)
endif()

add_library(mpc_protocol STATIC $<TARGET_OBJECTS:mpc_protocol_o>)

if (WITH_GPU)
    cuda_add_library(mpc_protocol_cu SHARED ${PROTO_SRCS_CU})
endif()

if (WITH_GRPC)
    target_link_libraries(mpc_protocol transport_o grpc++)
    cc_test(mesh_network_grpc_test SRCS network/mesh_network_grpc_test.cc DEPS mpc_protocol)
endif()

if (WITH_GPU)
    target_link_libraries(mpc_protocol ${GRPC_DEPS} gloo hiredis privc3 fluid_framework mpc_protocol_cu)
    target_link_libraries(mpc_protocol_cu ${GRPC_DEPS} gloo hiredis privc3 common_cu)
else ()
    target_link_libraries(mpc_protocol ${GRPC_DEPS} gloo hiredis privc3 privc_o fluid_framework)
endif()

cc_test(mesh_network_test SRCS network/mesh_network_test.cc DEPS mpc_protocol)
if (NOT WITH_GPU)
    cc_test(mpc_protocol_test SRCS mpc_protocol_test.cc DEPS mpc_protocol privc)
    cc_test(mpc_instance_test SRCS mpc_instance_test.cc DEPS mpc_protocol privc)
endif ()


