aux_source_directory(. DIR_SRCS)
aux_source_directory(./math MATH_SRCS)
add_library(mpc_ops_o OBJECT ${DIR_SRCS} ${MATH_SRCS})
add_dependencies(mpc_ops_o fluid_framework gloo)

add_library(mpc_ops STATIC $<TARGET_OBJECTS:mpc_ops_o>)
target_link_libraries(mpc_ops fluid_framework gloo)
