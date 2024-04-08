Alignment Pipeline for Finger Vein Matching
===========================================

The following instructions need to be executed in order to be able to use the finger vein matching algorithm written in C++. This for now is just an instruction for myself to ensure that I remember what I did.

## Installing Dependencies/Libraries 

I tried to use as little additional libraries as possible to first of all make this whole process simpler, but also being able to have an overview over everything that I'm using.

Now is maybe the time to flag, that this installation guide is written for Linux. If you use a different OS, you can have a look at the installation guides of each required library (links are provided below) to figure out what you have to do.

Generally you should have:
- gcc
- cmake
- git
- libpng

### NumCpp

This library is supposed to be a C++-version of NumPy, which, since the original code uses Python and makes heavy use of NumPy, hopefully avoids me and future developers to unnecessarily implement basic NumPy functions.

The installation process is as follows (found in [NumCpp Installation Guide][1]):

1. Clone the NumCpp repository from GitHub:

```sh
$ cd <the_directory_you_want_it_to_be_in>
$ git clone https://github.com/dpilger26/NumCpp.git
```

2. Build the install products using CMake:

```sh
$ cd NumCpp
$ mkdir build
$ cd build
$ cmake ..
```

3. Install the includes and CMake target files:

```sh
$ cmake --build .. --target install
```

> Note: Make sure that all dependencies like `cmake` and `libboost-all-dev`are installed.

### OpenCV

OpenCV is the next big library we will use. In combination with NumCpp, these two libraries contain almost all additional functions that we will need.

The installation guide for OpenCV can be found in [OpenCV Installation in Linux][2]:

1. Install all required packages. As of now, we will not need any of the optional packages:

```sh
$ sudo apt-get install build-essential
$ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

2. Obtain the OpenCV Source Code. I did this using their Git repository but feel free to directly download from their website, whatever you prefer:

```sh
$ cd <the_directory_you_want_it_to_be_in>
$ git clone https://github.com/opencv/opencv.git
```

3. Building OpenCV using CMake. In your directory should now be a folder named `opencv/`:

```sh
$ cd opencv/
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
$ make -j4 # runs 4 jobs in parallel
$ make install
```

If we want to use OpenCV to create a static library of our own, we additionally need to put the `-D BUILD_SHARED_LIBS=OFF` flag in the `cmake` command.

## Eigen
Eigen is needed for OpenCV, we will not use it for the project itself.

1. Download the latest stable version from Eigen Releases. I will use Eigen 3.4.0. And that's it. According to Eigen, all you have to do is include the header files and you can continue, but we in fact do still have to build it in order to use Eigen for our own CMake. In eigen-3.4.0/ do the following:

```sh
$ mkdir build
$ cd build
$ cmake ..
$ make install
```

## Building the Project

To build the project a `CMakeLists.txt` file needs to exist. To obtain an executable, execute the following commands in `alignment-cpp\`:

```sh
$ cmake -DOpenCV_DIR=./libraries/OpenCV_library/opencv/build -DEigen3_DIR=./libraries/Eigen_library/eigen-3.4.0 .
$ make -j4
```

If we want the intermediate steps to be saved, we need to uncomment Line 6 of our `CMakeLists.txt` file.

`./Cpp_alignment` then contains a calling example, to execute it type:

```sh
$ ./Cpp_alignment
```

This project also contains a testing and benchmarking infrastructure which can be accessed by uncommenting the needed lines in the `CMakeLists.txt` (Line 9-11) and executing:

```sh
$ ./Testbench
```



[1]: https://github.com/dpilger26/NumCpp/blob/master/docs/markdown/Installation.md
[2]: https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html
[3]: http://eigen.tuxfamily.org/index.php?title=Main_Page#Download