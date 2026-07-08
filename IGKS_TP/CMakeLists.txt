project(IGKS-TP LANGUAGES CXX CUDA)

file(GLOB_RECURSE SRC_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Simulator/*.cu"
)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
add_library(IGKS::TP ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC
  ${Python_INCLUDE_DIRS}
  ${NB_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  IGKS::TP::Misc
)

set(MODULE_NAME igks_tp)
nanobind_add_module(${MODULE_NAME} Binding.cu)
target_link_libraries(${MODULE_NAME} PRIVATE ${PROJECT_NAME})
