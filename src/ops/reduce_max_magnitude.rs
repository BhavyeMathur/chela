use crate::flat_index_generator::FlatIndexGenerator;
use crate::{IntegerDataType};
use num::{Bounded};
use crate::absolute::Absolute;
use crate::iterator::collapse_contiguous::has_uniform_stride;
use crate::util::partial_ord::{partial_max_magnitude};

pub(crate) trait ReduceMaxMagnitude: Copy + Absolute + Bounded + PartialOrd {
    /// Computes the max magnitude of `count` elements stored contiguously in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count` elements.
    unsafe fn max_magnitude_contiguous(ptr: *const Self, count: usize) -> Self {
        Self::max_magnitude_uniform_stride(ptr, count, 1)
    }

    /// Computes the max magnitude of `count` elements stored with a uniform stride in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count * stride` elements.
    unsafe fn max_magnitude_uniform_stride(mut ptr: *const Self, count: usize, stride: usize) -> Self {
        let mut output = Self::min_value();

        for _ in 0..count {
            output = partial_max_magnitude(*ptr, output);
            ptr = ptr.add(stride);
        }

        output
    }

    /// Computes the max magnitude of elements stored in a strided memory layout
    /// defined by `shape` and `stride` and pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must be a valid, non-null pointer to the memory region described by `shape` and `stride`.
    ///
    /// # Implementation
    /// - If the memory layout is contiguous, delegates this to the `max_contiguous()` function
    /// - If the memory layout has a uniform stride between elements, delegates to `max_uniform_stride()`
    /// - Otherwise, uses an unspecialized loop
    unsafe fn max_magnitude(ptr: *const Self, shape: &[usize], stride: &[usize]) -> Self {
        if let Some(stride) = has_uniform_stride(shape, stride) {
            return if stride == 1 {
                Self::max_magnitude_contiguous(ptr, shape.iter().product())
            } else {
                Self::max_magnitude_uniform_stride(ptr, shape.iter().product(), stride)
            };
        }

        let mut output = Self::min_value();
        for index in FlatIndexGenerator::from(shape, stride) {
            output = partial_max_magnitude(*ptr.add(index), output);
        }
        output
    }
}

impl<T: IntegerDataType> ReduceMaxMagnitude for T {}

impl ReduceMaxMagnitude for f32 {
    #[cfg(apple_vdsp)]
    unsafe fn max_magnitude_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use std::ptr::addr_of_mut;
        use crate::acceleration::vdsp::vDSP_maxmgv;
    
        let mut output = Self::min_value();
        unsafe { vDSP_maxmgv(ptr, stride as isize, addr_of_mut!(output), count); }
        output
    }
}

impl ReduceMaxMagnitude for f64 {
    #[cfg(apple_vdsp)]
    unsafe fn max_magnitude_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use std::ptr::addr_of_mut;
        use crate::acceleration::vdsp::vDSP_maxmgvD;
    
        let mut output = Self::min_value();
        unsafe { vDSP_maxmgvD(ptr, stride as isize, addr_of_mut!(output), count); }
        output
    }
}
