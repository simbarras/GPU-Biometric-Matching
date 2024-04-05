cmake_minimum_required(VERSION 3.20)
project( Cpp_alignment )
add_compile_options(-Wall -O3 -g)

# if you want the intermediate steps to be saved enable this option
# add_compile_definitions(SAVE_INTERMEDIATE_STEPS)

# set(CMAKE_C_FLAG_INIT "-fsanitize=address" "-fsanitize=undefined")
# set(CMAKE_CXX_FLAG_INIT "-fsanitize=address" "-fsanitize=undefined")

include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT ipoError)

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
    fingervein_extraction.cpp
    fingervein_extraction.h
    )

set(SOURCE_FILES ${pipeline_files})

# CMake instructions to make the static lib
add_library( alignmentCPP STATIC ${SOURCE_FILES})

if( ipoSupported )
    set_property(TARGET alignmentCPP PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

add_executable( Cpp_alignment main.c)

include_directories(${PNG_INCLUDE_DIR} "/usr/local/include/opencv4/")

find_package(NumCpp 2.12.1 REQUIRED)
target_link_libraries(Cpp_alignment
    NumCpp::NumCpp
)

set(OpenCV_STATIC ON)
find_package( OpenCV REQUIRED )
target_link_libraries( Cpp_alignment ${OpenCV_LIBS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (Cpp_alignment Eigen3::Eigen)

find_package(PNG REQUIRED)
target_link_libraries(Cpp_alignment ${PNG_LIBRARY})

target_link_libraries(Cpp_alignment alignmentCPP)
