set(VERILOGA_SOURCES
    lexer.cpp
    parser.cpp
    ast.cpp
    analyzer.cpp
    graph_assembler.cpp
)

add_library(acutesim_physics_veriloga OBJECT ${VERILOGA_SOURCES})
target_include_directories(acutesim_physics_veriloga PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(acutesim_physics_veriloga PRIVATE acutesim_components)
