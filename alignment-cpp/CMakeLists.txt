cmake_minimum_required(VERSION 3.20)
project( Cpp_alignment )
add_compile_options(-Wall -O3 -g)

# if you want the intermediate steps to be saved enable this option
# add_compile_definitions(SAVE_INTERMEDIATE_STEPS)

# if you want some timimg infrastructure to be enabled
# add_compile_definitions(BENCHMARK_PIPELINE)
# add_compile_definitions(BENCHMARK_PIPELINE_STEPS)
# add_compile_definitions(BENCHMARK_COMPARISON)

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
add_executable( Testbench benchmarkingAndTesting.cpp )

include_directories(${PNG_INCLUDE_DIR} "/usr/local/include/opencv4/")

find_package(NumCpp 2.12.1 REQUIRED)
target_link_libraries(alignmentCPP
    NumCpp::NumCpp
)

set(OpenCV_STATIC ON)
find_package( OpenCV REQUIRED )
target_link_libraries( alignmentCPP ${OpenCV_LIBS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (alignmentCPP Eigen3::Eigen)

find_package(PNG REQUIRED)
target_link_libraries(alignmentCPP ${PNG_LIBRARY})

target_link_libraries(Cpp_alignment alignmentCPP)
target_link_libraries(Testbench alignmentCPP)
