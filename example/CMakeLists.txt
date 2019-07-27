

LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/lib)

add_executable(main classification.cpp)

target_link_libraries(main
    ${OpenCV_LIBS}
    /home/tsui/yujc/caffe-ssd/build/lib/libcaffe.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
)
target_link_libraries(main "libsuperpoint.so")