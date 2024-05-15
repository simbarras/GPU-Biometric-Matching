Alignment Pipeline for Finger Vein Matching
===========================================

The following instructions need to be executed in order to be able to use the finger vein matching algorithm written in C++.

The first step when wanting to work with this project is of course to clone the repository:

```sh
$   git clone --recurse-submodules https://gitlab.epfl.ch/bioid/alignment.git
```

## Installing Dependencies/Libraries 

I tried to use as little additional libraries as possible to first of all make this whole process simpler, but also being able to have an overview over everything that I'm using.

Now is maybe the time to flag, that this installation guide is written for Linux. If you use a different OS, you can have a look at the installation guides of each required library (links are provided here [NumCpp Installation Guide][1], [OpenCV Installation in Linux][2], [Eigen][3]) to figure out what you have to do.

Generally you should have:
- gcc
- cmake
- git
- libpng
- libboost-all-dev

## Building the Project

You only have to execute the following lines and you are already good to go.

```sh
$   mkdir build/
$   cd build
$   cmake -DBUILD_SHARED_LIBS=OFF -DOpenCV_DIR=/usr/local/lib/opencv4/ ..
$   make -j4
```

If we want the intermediate steps to be saved, we need to uncomment Line 6 of our `CMakeLists.txt` file.

`./build/Cpp_alignment` then contains a callable example, to execute it type:

```sh
$   ./build/Cpp_alignment
```

## Profiling and Testing

This project also contains a testing and benchmarking infrastructure which can be accessed by executing the respective binaries generated through `cmake`. Execute any of the following lines to run the respective testing and/or benchmarking binary from `./alignment-cpp/`:

```sh
$   ./build/DistanceResults
$   ./build/TimePipComplete {start_index} {end_index}
$   ./build/TimeSteps {start_index} {end_index}
$   ./build/TimeMatchingSteps {start_index} {end_index}
```

If running the testbenches for the first time, you should make the directories first:
```sh
$   mkdir -p distance_results/after_optimizations/ timing/pipeline_complete/ timing/pipeline_steps/{edge_mask,prealignment,maximum_curvature,postalignment,distance}
```

`DistanceResults` runs the pipeline on each image and compares all the fingervein images. The results are written into two different files either `alignment-cpp/distance_results/{timestamp}distances_different_finger.csv`or `alignment-cpp/distance_results/{timestamp}distances_same_finger.csv` depending on whether the images belonged to the same finger or not. If Line 9 is uncommented in `CMakeLists.txt`then the results get written into `alignment-cpp/distance_results/after_optimizations/` instead.

`TimePipComplete` runs the entire pipeline several times on each dataset image in the range of `start_index` to `end_index` (exclusive), to obtain robust timing measurements. The results can be found in `alignment-cpp/timing/pipeline_complete/{timestamp}completePip.csv`.

`TimeSteps` runs the first three pipeline steps several times on each dataset image in the range of `start_index` to `end_index` (exclusive), to obtain robust timing measurements for each step respectively. The results can be found in `alignment-cpp/timing/pipeline_steps/{edge_mask, prealignment, maximum_curvature}`.

`TimeMatchingSteps` does logically the same thing as TimePipComplete but runs the postalignment and distance measure functions several times on each dataset image in the range of `start_index` to `end_index` (exclusive), to obtain robust timing measurements. The results can be found in `alignment-cpp/timing/pipeline_steps/{postalignment, distance}`.

If you do not want to run each Timing binary separately on a single core, you can also execute:

```sh
$   cd timing/
$   ./timing_runner.sh
```

which runs all timing executables one after another, but splits each run in 3 parts (each one mapped to a specific processor), the results are written in the same folders as running the executables manually. This of course assumes that you have 4 cores available, we leave one core such that the rest of the system does not interfere with the other 3 cores.

I also added an equivalent parallelization infrastructure to the Python implementation, which can be started in `alignment-python/timing/`.

### Unit Testing

While starting to optimize some functions, I added a unit test binary. You can add functions to this as you please. The code for this can be found in `alignment-cpp/unit_testing/unit_tests.cpp`, and the binary can be executed via:

```sh
$   ./build/UnitTesting
```

### FlameGraph

For profiling I added a submodule called FlameGraph to obtain a visualization of a stack trace analysis. FlameGraph can be used as follows, in the folder where your binary to profile is, execute:

```sh
$   sudo sysctl kernel.perf_event_paranoid=2
```

```sh
$   perf record -F max -g -o perf.data -- {your_binary}
$   perf script | ../FlameGraph/stackcollapse-perf.pl > /tmp/perf.folded
$   grep -F {the_name_of_the_function_you_want_to_profile} /tmp/perf.folded | ../FlameGraph/flamegraph.pl > flamegraph.svg
$   firefox flamegraph.svg &
```

[1]: https://github.com/dpilger26/NumCpp/blob/master/docs/markdown/Installation.md
[2]: https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html
[3]: http://eigen.tuxfamily.org/index.php?title=Main_Page#Download
