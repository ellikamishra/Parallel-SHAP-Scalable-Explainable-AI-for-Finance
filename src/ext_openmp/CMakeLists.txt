cmake_minimum_required(VERSION 3.18)
project(mc_shap_openmp LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_cmakedir
  RESULT_VARIABLE _pybind11_status
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(_pybind11_status EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
  list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
endif()
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  if(_pybind11_status EQUAL 0)
    message(FATAL_ERROR "pybind11Config.cmake not found under ${_pybind11_cmakedir}. Set pybind11_DIR or install pybind11 with CMake support.")
  else()
    message(FATAL_ERROR "pybind11 is not installed for interpreter ${Python3_EXECUTABLE}. Install it with 'pip install pybind11'.")
  endif()
endif()
find_package(OpenMP)
if(NOT TARGET OpenMP::OpenMP_CXX)
  if(APPLE)
    set(_libomp_roots
      /opt/homebrew/opt/libomp
      /usr/local/opt/libomp)
    foreach(_root IN LISTS _libomp_roots)
      if(EXISTS "${_root}/lib/libomp.dylib")
        message(STATUS "Configuring OpenMP manually using libomp at ${_root}")
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
          IMPORTED_LOCATION "${_root}/lib/libomp.dylib"
          INTERFACE_INCLUDE_DIRECTORIES "${_root}/include"
          INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
          INTERFACE_LINK_OPTIONS "-Xclang;-fopenmp"
          INTERFACE_LINK_LIBRARIES "${_root}/lib/libomp.dylib")
        set(_openmp_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _openmp_found)
      message(FATAL_ERROR "OpenMP runtime not found. Install it with 'brew install libomp' or provide OpenMP::OpenMP_CXX.")
    endif()
  else()
    message(FATAL_ERROR "OpenMP C++ support not found. Install an OpenMP-enabled toolchain.")
  endif()
endif()
pybind11_add_module(mc_shap_openmp mc_shap_openmp.cpp)
target_link_libraries(mc_shap_openmp PRIVATE OpenMP::OpenMP_CXX)
target_compile_features(mc_shap_openmp PRIVATE cxx_std_17)
