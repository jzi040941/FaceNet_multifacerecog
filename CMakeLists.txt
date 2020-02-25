cmake_minimum_required(VERSION 2.8)
project(main)

#set(QMAKE_CXXFLAGS "-std=c++11 -O3  -Wno-invalid-partial-specialization -Wno-sign-compare -w  -D_THREAD_SAFE -I/usr/local/include -L/usr/local/lib -lprotobuf -D_THREAD_SAFE")
#SET( CMAKE_CXX_FLAGS "-std=c++11 -O3  -Wno-invalid-partial-specialization -Wno-sign-compare -w  -D_THREAD_SAFE -I/usr/local/include -L/usr/local/lib -lprotobuf -D_THREAD_SAFE")

#SET( CMAKE_CXX_FLAGS "-std=c++11 -O3  -Wno-invalid-partial-specialization -Wno-sign-compare -w  -D_THREAD_SAFE -D_THREAD_SAFE")
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++11)


find_package( OpenCV REQUIRED)
message(STATUS "${OpenCV_INCLUDE_DIRS}") 

set(main_include

  /usr/local/opt/openblas/include 
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-genfiles 
  #${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen
  ${CMAKE_CURRENT_LIST_DIR}/eigen/
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/absl
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public

  ${CMAKE_CURRENT_LIST_DIR}/include/
  ${OpenCV_INCLUDE_DIRS}

)

set(main_src
  ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/mtcnn.cpp 
  ${CMAKE_CURRENT_LIST_DIR}/src/network.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/pBox.cpp


)
message("${main_include}")
  
  include_directories(
    ${main_include}
    )

#/usr/local/include/eigen3
#link_directories(/usr/local/opt/openblas/lib /usr/local/Cellar/opencv/3.3.1_1/lib)
#AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${main_src})
#target_link_libraries( main opencv_videoio opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs )
target_link_libraries( main openblas)
set(tensorflow_libs
  ${OpenCV_LIBS}

  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so.1
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so.1
  dl
  pthread
)
message("${tensorflow_libs}")
#target_link_libraries( main tensorflow_cc) 
target_link_libraries( main ${tensorflow_libs})
