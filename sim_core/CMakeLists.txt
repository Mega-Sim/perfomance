add_library(sim_core
  src/engine.cpp
)

target_include_directories(sim_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

add_executable(sim_run
  apps/sim_run.cpp
)

target_link_libraries(sim_run PRIVATE sim_core)

add_subdirectory(sim_core)
