
set(Caffe_INCLUDE_DIRS /home/tsui/yujc/caffe-ssd/include/ /usr/local/include /usr/local/cuda/include )
set(Caffe_LIBRARIES /home/tsui/yujc/caffe-ssd/build/lib/libcaffe.so
 /usr/lib/x86_64-linux-gnu/libglog.so
 /usr/lib/x86_64-linux-gnu/libboost_system.so)

find_package(OpenCV REQUIRED )
MESSAGE(STATUS "opencv: ${OpenCV_INCLUDE_DIRS}")

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
MESSAGE(STATUS "include : ${PROJECT_SOURCE_DIR}")
include_directories("${Caffe_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/include")

add_library(exampleapp SHARED superpoint.cpp)
target_link_libraries(exampleapp "${OpenCV_LIBRARIES}")
target_link_libraries(exampleapp "${Caffe_LIBRARIES}" )
set_property(TARGET exampleapp PROPERTY CXX_STANDARD 11)


#add_executable(main src/classification.cpp)
#target_link_libraries(main "${OpenCV_LIBRARIES}")
#target_link_libraries(main "libexampleapp.so")
#set_property(TARGET main PROPERTY CXX_STANDARD 11)
