cmake_minimum_required(VERSION 3.20)
project( Cpp_alignment )

add_executable( Cpp_alignment pipeline.cpp )

find_package(NumCpp 2.12.1 REQUIRED)
target_link_libraries(Cpp_alignment
    NumCpp::NumCpp
)

find_package( OpenCV REQUIRED )
target_link_libraries( Cpp_alignment ${OpenCV_LIBS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (Cpp_alignment Eigen3::Eigen)