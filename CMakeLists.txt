cmake_minimum_required (VERSION 3.5)
project (WST-DL)

set (CMAKE_CXX_STANDARD 11)

add_executable(test_conv test_conv.cc)

set(USING_INTEL FALSE)
if (${CMAKE_CXX_COMPILER} MATCHES "icpc.*$")
  set(USING_INTEL TRUE)
endif()

if (USING_INTEL)
  target_compile_options(test_conv PUBLIC "-mkl")
endif(USING_INTEL)

if (USING_INTEL)
  set_target_properties(test_conv PROPERTIES LINK_FLAGS "-mkl")
endif(USING_INTEL)

