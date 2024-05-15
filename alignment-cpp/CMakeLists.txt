cmake_minimum_required(VERSION 3.20)
project( Cpp_alignment )
add_compile_options(-Wall -O3 -g -fno-omit-frame-pointer)

# if you want the intermediate steps to be saved enable this option
# add_compile_definitions(SAVE_INTERMEDIATE_STEPS)

# if you want some timimg infrastructure to be enabled
# add_compile_definitions(WITH_OPT)

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
add_executable( UnitTesting unit_testing/unit_tests.cpp)
add_executable( DistanceResults distance_results/distance_results_for_analysis.cpp)
add_executable( TimePipComplete timing/pipeline_complete/measureTime_pipelineComplete.cpp)
add_executable( TimeSteps timing/pipeline_steps/measureTime_pipelineSteps.cpp)
add_executable( TimeMatchingSteps timing/pipeline_steps/measureTime_matchingSteps.cpp)

add_subdirectory(libraries/NumCpp)
target_link_libraries(alignmentCPP
    NumCpp::NumCpp
)

add_subdirectory(libraries/eigen)
target_link_libraries (alignmentCPP Eigen3::Eigen)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( alignmentCPP ${OpenCV_LIBS} )

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(alignmentCPP ${PNG_LIBRARY})

set(ALIGNMENT_CPP alignmentCPP)
target_link_libraries(Cpp_alignment PUBLIC ${ALIGNMENT_CPP})
target_link_libraries(DistanceResults PUBLIC ${ALIGNMENT_CPP})
target_link_libraries(TimePipComplete PUBLIC ${ALIGNMENT_CPP})
target_link_libraries(TimeSteps PUBLIC ${ALIGNMENT_CPP})
target_link_libraries(TimeMatchingSteps PUBLIC ${ALIGNMENT_CPP})
target_link_libraries(UnitTesting PUBLIC ${ALIGNMENT_CPP})
