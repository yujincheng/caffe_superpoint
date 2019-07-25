cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)


#SET( CMAKE_VERBOSE_MAKEFILE ON )
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(example)


#add_executable(main src/classification.cpp)
#target_link_libraries(main "${OpenCV_LIBRARIES}")
#target_link_libraries(main "libexampleapp.so")
#set_property(TARGET main PROPERTY CXX_STANDARD 11)
