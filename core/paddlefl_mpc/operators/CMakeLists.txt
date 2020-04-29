add_compile_options(-msse4.2 -maes)

aux_source_directory(. DIR_SRCS)
add_library(mpc_ops_o OBJECT ${DIR_SRCS})
add_dependencies(mpc_ops_o fluid_framework gloo)

add_library(mpc_ops STATIC $<TARGET_OBJECTS:mpc_ops_o>)
target_link_libraries(mpc_ops fluid_framework gloo)
