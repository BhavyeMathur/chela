use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::has_uniform_stride;
use crate::IntegerDataType;
use num::{One, Zero};
use std::ops::{AddAssign, MulAssign};


pub(crate) trait ReduceProduct: Zero + One + Copy + MulAssign + AddAssign {
    /// Computes the product of `count` elements stored contiguously in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count` elements.
    unsafe fn product_contiguous(ptr: *const Self, count: usize) -> Self {
        Self::product_uniform_stride(ptr, count, 1)
    }

    /// Computes the product of `count` elements stored with a uniform stride in memory pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid array of `count * stride` elements.
    unsafe fn product_uniform_stride(mut ptr: *const Self, count: usize, stride: usize) -> Self {
        let mut output = Self::one();

        for _ in 0..count {
            output *= *ptr;
            ptr = ptr.add(stride);
        }

        output
    }

    /// Computes the product of elements stored in a strided memory layout
    /// defined by `shape` and `stride` and pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must be a valid, non-null pointer to the memory region described by `shape` and `stride`.
    ///
    /// # Implementation
    /// - If the memory layout is contiguous, delegates this to the `product_contiguous()` function
    /// - If the memory layout has a uniform stride between elements, delegates to `product_uniform_stride()`
    /// - Otherwise, uses an unspecialized loop
    unsafe fn product(ptr: *const Self, shape: &[usize], stride: &[usize]) -> Self {
        if let Some(stride) = has_uniform_stride(shape, stride) {
            return if stride == 1 {
                Self::product_contiguous(ptr, shape.iter().product())
            } else {
                Self::product_uniform_stride(ptr, shape.iter().product(), stride)
            };
        }

        let mut output = Self::one();
        for index in FlatIndexGenerator::from(shape, stride) {
            output *= *ptr.add(index);
        }
        output
    }
}

impl<T: IntegerDataType> ReduceProduct for T {}

impl ReduceProduct for f32 {
    #[cfg(neon_simd)]
    unsafe fn product_contiguous(ptr: *const Self, count: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_product_contiguous(ptr, count)
    }

    #[cfg(neon_simd)]
    unsafe fn product_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_product_uniform(ptr, count, stride)
    }
}

impl ReduceProduct for f64 {
    #[cfg(neon_simd)]
    unsafe fn product_contiguous(ptr: *const Self, count: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_product_contiguous(ptr, count)
    }

    #[cfg(neon_simd)]
    unsafe fn product_uniform_stride(ptr: *const Self, count: usize, stride: usize) -> Self {
        use crate::ops::simd_reduce_ops::SIMDReduceOps;
        Self::simd_product_uniform(ptr, count, stride)
    }
}
