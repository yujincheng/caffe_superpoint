# caffe_superpoint

## directory

''src'' includes  the  source file to build library ''libexampleapp.so''
''include'' includes the header file when running examples
''example'' includes a superpoint extraction example
''demo2'' includes the caffe model

## built caffe dir

Target the the Caffe_include_dir and Caffe_library in ''src/CMakeLists.txt'' ''example/CMakeLists.txt''

## build and run

```
mkdir build
cd build
cmake ..
make
./example/main
```
