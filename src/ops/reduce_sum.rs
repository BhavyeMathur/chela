use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::has_uniform_stride;
use crate::IntegerDataType;
use num::Zero;
use std::ops::AddAssign;


pub(crate) trait ReduceSum: Zero + Copy + AddAssign {
    /// Computes the sum of `count` elements stored contiguously in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count` elements.
    unsafe fn sum_contiguous(ptr: *const Self, count: usize) -> Self {
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

impl<T: IntegerDataType> ReduceSum for T {}

impl ReduceSum for f32 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sum_contiguous(ptr: *const Self, count: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_sum_contiguous(ptr, count)
    }

    #[cfg(apple_vdsp)]
    unsafe fn sum_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use std::ptr::addr_of_mut;
        use crate::acceleration::vdsp::vDSP_sve;
    
        let mut output = Self::zero();
        unsafe { vDSP_sve(ptr, stride as isize, addr_of_mut!(output), count); }
        output
    }
    
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sum_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_sum_uniform(ptr, count, stride)
    }
}

impl ReduceSum for f64 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sum_contiguous(ptr: *const Self, count: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_sum_contiguous(ptr, count)
    }

    #[cfg(apple_vdsp)]
    unsafe fn sum_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use std::ptr::addr_of_mut;
        use crate::acceleration::vdsp::vDSP_sveD;
    
        let mut output = Self::zero();
        unsafe { vDSP_sveD(ptr, stride as isize, addr_of_mut!(output), count); }
        output
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sum_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_sum_uniform(ptr, count, stride)
    }
}
