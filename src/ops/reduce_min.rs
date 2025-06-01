use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::has_uniform_stride;
use crate::{IntegerDataType};
use num::{Bounded};
use crate::util::partial_ord::partial_min;

pub(crate) trait ReduceMin: Copy + PartialOrd + Bounded {
    /// Computes the min of `count` elements stored contiguously in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count` elements.
    unsafe fn min_contiguous(ptr: *const Self, count: usize) -> Self {
        Self::min_uniform_stride(ptr, count, 1)
    }

    /// Computes the min of `count` elements stored with a uniform stride in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count * stride` elements.
    unsafe fn min_uniform_stride(mut ptr: *const Self, count: usize, stride: usize) -> Self {
        let mut output = Self::max_value();

        for _ in 0..count {
            output = partial_min(*ptr, output);
            ptr = ptr.add(stride);
        }

        output
    }

    /// Computes the min of elements stored in a strided memory layout
    /// defined by `shape` and `stride` and pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must be a valid, non-null pointer to the memory region described by `shape` and `stride`.
    ///
    /// # Implementation
    /// - If the memory layout is contiguous, delegates this to the `min_contiguous()` function
    /// - If the memory layout has a uniform stride between elements, delegates to `min_uniform_stride()`
    /// - Otherwise, uses an unspecialized loop
    unsafe fn min(ptr: *const Self, shape: &[usize], stride: &[usize]) -> Self {
        if let Some(stride) = has_uniform_stride(shape, stride) {
            return if stride == 1 {
                Self::min_contiguous(ptr, shape.iter().product())
            } else {
                Self::min_uniform_stride(ptr, shape.iter().product(), stride)
            };
        }

        let mut output = Self::max_value();
        for index in FlatIndexGenerator::from(shape, stride) {
            output = partial_min(*ptr.add(index), output);
        }
        output
    }
}

impl<T: IntegerDataType> ReduceMin for T {}

impl ReduceMin for f32 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn min_contiguous(ptr: *const Self, count: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_min_contiguous(ptr, count)
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn min_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use std::ptr::addr_of_mut;
        use crate::acceleration::vdsp::vDSP_minv;
    
        let mut output = Self::max_value();
        unsafe { vDSP_minv(ptr, stride as isize, addr_of_mut!(output), count as isize); }
        output
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn min_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_min_uniform(ptr, count, stride)
    }
}

impl ReduceMin for f64 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn min_contiguous(ptr: *const Self, count: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_min_contiguous(ptr, count)
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn min_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use std::ptr::addr_of_mut;
        use crate::acceleration::vdsp::vDSP_minvD;
    
        let mut output = Self::max_value();
        unsafe { vDSP_minvD(ptr, stride as isize, addr_of_mut!(output), count as isize); }
        output
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn min_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_min_uniform(ptr, count, stride)
    }
}
