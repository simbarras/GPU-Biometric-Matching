cmake_minimum_required(VERSION 3.20)
project( Cpp_alignment )

set(pipeline_files
    distance.cpp 
    distance.hpp
    extraction.cpp 
    extraction.hpp
    helper.hpp
    mask_extraction.cpp 
    mask_extraction.hpp
    pipeline.cpp 
    pipeline.hpp
    postalignment.cpp 
    postalignment.hpp
    prealignment.cpp 
    prealignment.hpp
    )

set(SOURCE_FILES fingervein_extraction.cpp ${pipeline_files})

add_executable( Cpp_alignment ${SOURCE_FILES} )

find_package(NumCpp 2.12.1 REQUIRED)
target_link_libraries(Cpp_alignment
    NumCpp::NumCpp
)

find_package( OpenCV REQUIRED )
target_link_libraries( Cpp_alignment ${OpenCV_LIBS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (Cpp_alignment Eigen3::Eigen)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(Cpp_alignment ${PNG_LIBRARY})
