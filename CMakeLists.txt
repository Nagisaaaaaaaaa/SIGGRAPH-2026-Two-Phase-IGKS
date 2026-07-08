cmake_minimum_required(VERSION 3.25.2)
project(SIGGRAPH-2026-Two-Phase-IGKS LANGUAGES CXX CUDA)

set(IGKS_TP_BINARY_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory for IGKS-TP binary outputs.")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${IGKS_TP_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${IGKS_TP_BINARY_DIR}")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

execute_process(
  COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
  OUTPUT_VARIABLE IGKS_TP_CUDA_COMPUTE_CAPS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCHALL "[0-9]+\\.[0-9]+" IGKS_TP_CUDA_COMPUTE_CAPS "${IGKS_TP_CUDA_COMPUTE_CAPS}")
list(SORT IGKS_TP_CUDA_COMPUTE_CAPS COMPARE NATURAL ORDER DESCENDING)
list(GET IGKS_TP_CUDA_COMPUTE_CAPS 0 IGKS_TP_CUDA_COMPUTE_CAP)
string(REPLACE "." "" IGKS_TP_CUDA_ARCH "${IGKS_TP_CUDA_COMPUTE_CAP}")
set(CMAKE_CUDA_ARCHITECTURES "${IGKS_TP_CUDA_ARCH}-real")
message(STATUS "IGKS-TP: CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

include(FetchContent)
FetchContent_Declare(
  CPM
  GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
  GIT_TAG v0.42.0
)
FetchContent_MakeAvailable(CPM)

CPMAddPackage(
  NAME arietta
  GITHUB_REPOSITORY Nagisaaaaaaaaa/arietta
  GIT_TAG main
  OPTIONS "ARIETTA_ENABLE_TESTING OFF"
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  RESULT_VARIABLE IGKS_TP_NANOBIND_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT IGKS_TP_NANOBIND_RESULT EQUAL 0)
  message(FATAL_ERROR "IGKS-TP: failed to query nanobind CMake directory from ${Python_EXECUTABLE}")
endif()

find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_DIR}" NO_DEFAULT_PATH)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Misc/Binding.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/Misc/Binding.py"
  @ONLY
)

add_subdirectory(IGKS_TP)
add_subdirectory(Misc)
