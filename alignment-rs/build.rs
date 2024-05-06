use cmake::Config;

fn main() {
    println!("cargo::rerun-if-changed=../alignment-cpp");

    let dst_opencv = Config::new("../alignment-cpp/libraries/opencv")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_PERF_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_opencv_apps", "OFF")
        .build();

    let dst = Config::new("../alignment-cpp")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("NUMCPP_NO_USE_BOOST", "ON")
        .define("OpenCV_DIR", dst_opencv.display().to_string())
        .build_target("alignmentCPP")
        .build();

    println!("cargo::rustc-link-search=native={}/lib", dst.display());
    println!(
        "cargo::rustc-link-search=native={}/lib/opencv4/3rdparty",
        dst.display()
    );
    println!("cargo::rustc-link-search=native={}/build", dst.display());
    println!("cargo::rustc-link-lib=static=alignmentCPP");
    println!("cargo::rustc-link-lib=static=opencv_core");
    println!("cargo::rustc-link-lib=static=opencv_imgproc");
    println!("cargo::rustc-link-lib=static=opencv_imgcodecs");
    println!("cargo::rustc-link-lib=static=ippicv");
    println!("cargo::rustc-link-lib=static=ippiw");
    println!("cargo::rustc-link-lib=static=ittnotify");
    println!("cargo::rustc-link-lib=dylib=stdc++");
}
