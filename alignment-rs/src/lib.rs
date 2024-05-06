use std::fmt::Debug;

mod sys;

extern crate core;

/// Represents the perspective from which an image was taken when looking at the
/// vein scanner head-on from the direction from which the finger is inserted
/// (you'll see the logo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraPerspective {
    Left = 1,
    Right,
}

impl CameraPerspective {
    fn get_u8(self) -> u8 {
        match self {
            CameraPerspective::Left => 1,
            CameraPerspective::Right => 2,
        }
    }
}

/// Represents a model of a single fingervein perspective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSingle {
    model: Vec<u8>,
}

impl ModelSingle {
    /// Returns a byte-serialisation of the model may not be modified
    /// externally.
    pub fn as_bytes(&self) -> &[u8] {
        &self.model
    }

    /// Creates a model from raw model bytes.
    ///
    /// ## Safety
    /// The input bytes MUST have previously come from a call to
    /// `ModelSingle::as_bytes()`.
    pub fn from_bytes_unchecked(model_single: &[u8]) -> Self {
        Self {
            model: model_single.to_vec(),
        }
    }
}

/// Represents a model of a finger's fingerveins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    model: Vec<u8>,
}

impl Model {
    /// Returns a byte-serialisation of the model may not be modified
    /// externally.
    pub fn as_bytes(&self) -> &[u8] {
        &self.model
    }

    /// Creates a model from raw model bytes.
    ///
    /// ## Safety
    /// The input bytes MUST have previously come from a call to
    /// `Model::as_bytes()`.
    pub fn from_bytes_unchecked(model_single: &[u8]) -> Self {
        Self {
            model: model_single.to_vec(),
        }
    }
}

/// A cache for fingervein probes to avoid repeated computation
struct ProbeCache {
    cache: *mut sys::probeCache,
}

impl Debug for ProbeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProbeCache").finish()
    }
}

impl Drop for ProbeCache {
    fn drop(&mut self) {
        unsafe { sys::freeProbeCache(self.cache) };
    }
}

impl ProbeCache {
    pub fn new() -> ProbeCache {
        ProbeCache {
            cache: unsafe { sys::new_probeCache() },
        }
    }
}

impl Default for ProbeCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ImageModelComparatorSingle {
    perspective: CameraPerspective,
    width: usize,
    grayscale_image: Vec<u8>,
    cache: ProbeCache,
}

impl ImageModelComparatorSingle {
    /// Creates a new model comparator for a given input image. May be used for
    /// mass comparison.
    ///
    /// The `grayscale_image` contains one 8-bit luminosity value per pixel.
    ///
    /// ## Panics
    /// If the grayscale image is not rectangular when interpreted with the
    /// given `width`, or if the image size is larger that 2^31 x 2^31.
    pub fn new(
        perspective: CameraPerspective,
        width: usize,
        grayscale_image: &[u8],
    ) -> ImageModelComparatorSingle {
        let height = grayscale_image.len() / width;
        assert!(height * width == grayscale_image.len());
        assert!(width < (1 << 31) && height < (1 << 31));

        ImageModelComparatorSingle {
            perspective,
            width,
            grayscale_image: grayscale_image.to_vec(),
            cache: ProbeCache::new(),
        }
    }

    /// Compares the image given to `::new()` with the given `model`.
    ///
    /// Returns true if it matches, false otherwise.
    pub fn compare_with_model(&self, model: &ModelSingle) -> bool {
        let height = self.grayscale_image.len() / self.width;

        unsafe {
            sys::compare_model_with_input_single(
                self.width.try_into().unwrap(),
                height.try_into().unwrap(),
                self.perspective.get_u8().into(),
                self.grayscale_image.as_ptr(),
                model.model.as_ptr(),
                model.model.len(),
                self.cache.cache,
            )
        }
    }
}

#[derive(Debug)]
pub struct ImageModelComparator {
    tau: f64,
    width: usize,
    left_image: Vec<u8>,
    right_image: Vec<u8>,
    cache: ProbeCache,
}

impl ImageModelComparator {
    /// Creates a new model comparator for a given input image. May be used for
    /// mass comparison.
    ///
    /// `tau` is a value chosen to minimise equal error rate.
    /// `left_image` and `right_image` each contain one 8-bit luminosity value per pixel.
    ///
    /// ## Panics
    /// If the grayscale images are not rectangular when interpreted with the given
    /// `width`, or if the image size is larger that 2^31 x 2^31.
    pub fn new(
        tau: f64,
        width: usize,
        left_image: &[u8],
        right_image: &[u8],
    ) -> ImageModelComparator {
        let height = left_image.len() / width;
        assert!(height * width == left_image.len());
        assert!(height * width == right_image.len());
        assert!(width < (1 << 31) && height < (1 << 31));

        ImageModelComparator {
            tau,
            width,
            left_image: left_image.to_vec(),
            right_image: right_image.to_vec(),
            cache: ProbeCache::new(),
        }
    }

    /// Compares the images given to `::new()` with the given `model`.
    ///
    /// Returns true if it matches, false otherwise.
    pub fn compare_with_model(&self, model: &Model) -> bool {
        let height = self.left_image.len() / self.width;

        unsafe {
            sys::compare_model_with_input(
                self.width.try_into().unwrap(),
                height.try_into().unwrap(),
                self.tau,
                self.left_image.as_ptr(),
                self.right_image.as_ptr(),
                model.model.as_ptr(),
                model.model.len(),
                self.cache.cache,
            )
        }
    }
}

/// Generate a `ModelSingle` from a single fingervein image.
///
/// The `grayscale_image` contains one 8-bit luminosity value per pixel.
///
/// ## Panics
/// If the grayscale image is not rectangular when interpreted with the given
/// `width`, or if the image size is larger that 2^31 x 2^31.
pub fn register_fingervein_single(
    perspective: CameraPerspective,
    width: usize,
    grayscale_image: &[u8],
) -> ModelSingle {
    let height = grayscale_image.len() / width;
    assert!(height * width == grayscale_image.len());

    let mut model_ptr = core::ptr::null_mut();
    let model_ptr_ptr = &mut model_ptr;

    let model_sz = unsafe {
        sys::register_fingervein_single(
            width.try_into().unwrap(),
            height.try_into().unwrap(),
            perspective.get_u8().into(),
            model_ptr_ptr,
            grayscale_image.as_ptr(),
        )
    };

    let slice = unsafe { core::slice::from_raw_parts(model_ptr, model_sz) };

    let model = slice.to_vec();

    unsafe {
        sys::free_model(model_ptr);
    }

    ModelSingle { model }
}

/// Generate a `Model` from a two fingervein images.
///
/// `left_image` and `right_image` each contain one 8-bit luminosity value per pixel.
///
/// ## Panics
/// If the grayscale images are not rectangular when interpreted with the given
/// `width`, or if the image size is larger that 2^31 x 2^31.
pub fn register_fingerveins(width: usize, left_image: &[u8], right_image: &[u8]) -> Model {
    let height = left_image.len() / width;
    assert!(height * width == left_image.len());
    assert!(height * width == right_image.len());

    let mut model_ptr = core::ptr::null_mut();
    let model_ptr_ptr = &mut model_ptr;

    let model_sz = unsafe {
        sys::register_fingerveins(
            width.try_into().unwrap(),
            height.try_into().unwrap(),
            model_ptr_ptr,
            left_image.as_ptr(),
            right_image.as_ptr(),
        )
    };

    let slice = unsafe { core::slice::from_raw_parts(model_ptr, model_sz) };

    let model = slice.to_vec();

    unsafe {
        sys::free_model(model_ptr);
    }

    Model { model }
}

#[cfg(test)]
mod tests {
    use crate::{
        register_fingervein_single, register_fingerveins, CameraPerspective, ImageModelComparator,
        ImageModelComparatorSingle,
    };

    #[test]
    fn simple() {
        // 376 x 240
        let mut pseudo_data = vec![0u8; 376 * 240];

        for i in 0..(240 * 376) {
            pseudo_data[i] = (i % 256) as u8;
        }

        let model = register_fingervein_single(CameraPerspective::Left, 376, &pseudo_data);

        let comp = ImageModelComparatorSingle::new(CameraPerspective::Left, 376, &pseudo_data);

        assert!(comp.compare_with_model(&model));
    }

    #[test]
    fn basic() {
        let mut pseudo_data = vec![0u8; 376 * 240];
        let mut pseudo_data_2 = vec![0u8; 376 * 240];

        for i in 0..(240 * 376) {
            pseudo_data[i] = (i % 256) as u8;
        }

        for i in 0..(240 * 376) {
            pseudo_data_2[i] = (255 - i % 256) as u8;
        }

        let model = register_fingerveins(376, &pseudo_data, &pseudo_data_2);

        let comp = ImageModelComparator::new(0.55, 376, &pseudo_data, &pseudo_data_2);

        assert!(comp.compare_with_model(&model))
    }
}
