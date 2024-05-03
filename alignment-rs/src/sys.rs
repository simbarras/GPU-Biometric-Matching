use libc::size_t;
use std::ffi::{c_double, c_int};

#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct probeCache {
    _private: [u8; 0],
}

#[link(name = "alignmentCPP")]
extern "C" {
    pub(crate) fn register_fingervein_single(
        width: c_int,
        height: c_int,
        camera_perspective: c_int,
        modelOut: *mut *mut u8,
        imageIn: *const u8,
    ) -> size_t;

    pub(crate) fn register_fingerveins(
        width: c_int,
        height: c_int,
        modelOut: *mut *mut u8,
        imageIn1: *const u8,
        imageIn2: *const u8,
    ) -> size_t;

    pub(crate) fn new_probeCache() -> *mut probeCache;

    pub(crate) fn freeProbeCache(cache: *mut probeCache);

    pub(crate) fn compare_model_with_input_single(
        width: c_int,
        height: c_int,
        camera_perspective: c_int,
        imageIn: *const u8,
        modelIn: *const u8,
        modelSize: size_t,
        probeC: *mut probeCache,
    ) -> bool;

    pub(crate) fn compare_model_with_input(
        width: c_int,
        height: c_int,
        tau: c_double,
        imageIn1: *const u8,
        imageIn2: *const u8,
        modelIn: *const u8,
        modelSize: size_t,
        probeC: *mut probeCache,
    ) -> bool;

    pub(crate) fn free_model(model: *mut u8);
}
