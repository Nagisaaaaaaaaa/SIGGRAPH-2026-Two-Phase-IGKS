project(IGKSTP-Misc LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} INTERFACE)
add_library(IGKSTP::Misc ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(${PROJECT_NAME} INTERFACE
  Arietta
  CUDA::cudart
)

if(MSVC)
  target_compile_options(${PROJECT_NAME} INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
  )
endif()
