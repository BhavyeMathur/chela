#![allow(unused_mut)]
#![allow(unused_variables)]

use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::has_uniform_stride;
use crate::IntegerDataType;
use num::{One, Zero};
use std::ops::{Add, AddAssign};

#[cfg(apple_vdsp)]
use std::ptr::addr_of_mut;

#[cfg(apple_vdsp)]
use crate::accelerate::vdsp::*;


pub(crate) trait Reduce: Zero + One + Copy + Add<Output=Self> + AddAssign {
    /// Computes the sum of `count` elements stored contiguously in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count` elements.
    unsafe fn sum_contiguous(mut ptr: *const Self, count: usize) -> Self {
        Self::sum_uniform_stride(ptr, count, 1)
    }

    /// Computes the sum of `count` elements stored with a uniform stride in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count * stride` elements.
    unsafe fn sum_uniform_stride(mut ptr: *const Self, count: usize, stride: usize) -> Self {
        let mut output = Self::zero();

        for _ in 0..count {
            output += *ptr;
            ptr = ptr.add(stride);
        }

        output
    }

    /// Computes the sum of elements stored in a strided memory layout
    /// defined by `shape` and `stride` and pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must be a valid, non-null pointer to the memory region described by `shape` and `stride`.
    ///
    /// # Implementation
    /// - If the memory layout is contiguous, delegates this to the `sum_contiguous()` function
    /// - If the memory layout has a uniform stride between elements, delegates to `sum_uniform_stride()`
    /// - Otherwise, uses an unspecialized loop
    unsafe fn sum(ptr: *const Self, shape: &[usize], stride: &[usize]) -> Self {
        if let Some(stride) = has_uniform_stride(shape, stride) {
            return if stride == 1 {
                Self::sum_contiguous(ptr, shape.iter().product())
            } else {
                Self::sum_uniform_stride(ptr, shape.iter().product(), stride)
            };
        }

        let mut output = Self::zero();
        for index in FlatIndexGenerator::from(shape, stride) {
            output += *ptr.add(index);
        }
        output
    }
}

impl<T: IntegerDataType> Reduce for T {}

impl Reduce for f32 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sum_contiguous(mut ptr: *const Self, mut count: usize) -> Self {
        use crate::accelerate::simd::SIMD;
        Self::simd_sum_contiguous(ptr, count)
    }

    #[cfg(apple_vdsp)]
    unsafe fn sum_uniform_stride(mut ptr: *const Self, count: usize, stride: usize) -> Self {
        let mut output = Self::zero();
        unsafe { vDSP_sve(ptr, stride as isize, addr_of_mut!(output), count as isize); }
        output
    }
}

impl Reduce for f64 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sum_contiguous(mut ptr: *const Self, mut count: usize) -> Self {
        use crate::accelerate::simd::SIMD;
        Self::simd_sum_contiguous(ptr, count)
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn sum_uniform_stride(mut ptr: *const Self, count: usize, stride: usize) -> Self {
        let mut output = Self::zero();
        unsafe { vDSP_sveD(ptr, stride as isize, addr_of_mut!(output), count as isize); }
        output
    }
}
