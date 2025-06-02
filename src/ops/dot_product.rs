use crate::IntegerDataType;
use rand_distr::num_traits::Zero;
use std::ops::{AddAssign, Mul};

pub(crate) trait DotProduct: AddAssign + Mul<Output=Self> + Zero + Copy {
    /// Performs a dot product between `count` elements beginning at `src0` and `src1`
    /// and writing the result to `dst`
    ///
    /// # Safety
    /// - `src0` and `src1` must represent a valid array of `count` elements.
    /// - `dst` must be a valid pointer
    unsafe fn dot_product(mut src0: *const Self, mut src1: *const Self, dst: *mut Self, count: usize) {
        let mut sum = Self::zero();

        for _ in 0..count {
            sum += (*src0) * (*src1);
            src0 = src0.add(1);
            src1 = src1.add(1);
        }

        *dst += sum;
    }

    /// Performs a dot product between `count` elements beginning at `src0` and `src1`
    /// with a space of `stride0` and `stride1` between elements respectively,
    /// and writing the result to `dst`
    ///
    /// # Safety
    /// - `src0` must point to a valid array of `count * stride0` elements.
    /// - `src1` must point to a valid array of `count * stride1` elements.
    /// - `dst` must be a valid pointer
    unsafe fn strided_dot_product(mut src0: *const Self, stride0: usize,
                                  mut src1: *const Self, stride1: usize,
                                  dst: *mut Self, count: usize) {
        let mut sum = Self::zero();

        for _ in 0..count {
            sum += (*src0) * (*src1);
            src0 = src0.add(stride0);
            src1 = src1.add(stride1);
        }

        *dst += sum;
    }
}

impl<T: IntegerDataType> DotProduct for T {}

impl DotProduct for f32 {
    #[cfg(apple_vdsp)]
    unsafe fn dot_product(src0: *const Self, src1: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_dotpr;
        use std::ptr::addr_of_mut;

        let mut sum = Self::zero();
        vDSP_dotpr(src0, 1, src1, 1, addr_of_mut!(sum), count);
        *dst += sum;
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn dot_product(src0: *const Self, src1: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_sum_of_products::SIMDSumOfProducts;
        Self::simd_dot_product(src0, src1, dst, count);
    }

    #[cfg(all(not(apple_vdsp), not(neon_simd), blas))]
    unsafe fn dot_product(src0: *const Self, src1: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::cblas::cblas_sdot;
        *dst += cblas_sdot(count as i32, src0, 1, src1, 1);
    }


    #[cfg(apple_vdsp)]
    unsafe fn strided_dot_product(src0: *const Self, stride0: usize,
                                  src1: *const Self, stride1: usize,
                                  dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_dotpr;
        use std::ptr::addr_of_mut;

        let mut sum = Self::zero();
        vDSP_dotpr(src0, stride0 as isize, src1, stride1 as isize, addr_of_mut!(sum), count);
        *dst += sum;
    }

    #[cfg(all(not(apple_vdsp), not(neon_simd), blas))]
    unsafe fn strided_dot_product(src0: *const Self, stride0: usize,
                                  src1: *const Self, stride1: usize,
                                  dst: *mut Self, count: usize) {
        use crate::acceleration::cblas::cblas_sdot;
        *dst += cblas_sdot(count as i32, src0, stride0 as i32, src1, stride1 as i32);
    }
}

impl DotProduct for f64 {
    #[cfg(apple_vdsp)]
    unsafe fn dot_product(src0: *const Self, src1: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_dotprD;
        use std::ptr::addr_of_mut;

        let mut sum = Self::zero();
        vDSP_dotprD(src0, 1, src1, 1, addr_of_mut!(sum), count);
        *dst += sum;
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn dot_product(src0: *const Self, src1: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_sum_of_products::SIMDSumOfProducts;
        Self::simd_dot_product(src0, src1, dst, count);
    }

    #[cfg(all(not(apple_vdsp), not(neon_simd), blas))]
    unsafe fn dot_product(src0: *const Self, src1: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::cblas::cblas_ddot;
        *dst += cblas_ddot(count as i32, src0, 1, src1, 1);
    }

    #[cfg(apple_vdsp)]
    unsafe fn strided_dot_product(src0: *const Self, stride0: usize,
                                  src1: *const Self, stride1: usize,
                                  dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_dotprD;
        use std::ptr::addr_of_mut;

        let mut sum = Self::zero();
        vDSP_dotprD(src0, stride0 as isize, src1, stride1 as isize, addr_of_mut!(sum), count);
        *dst += sum;
    }

    #[cfg(all(not(apple_vdsp), blas))]
    unsafe fn strided_dot_product(src0: *const Self, stride0: usize,
                                  src1: *const Self, stride1: usize,
                                  dst: *mut Self, count: usize) {
        use crate::acceleration::cblas::cblas_ddot;
        *dst += cblas_ddot(count as i32, src0, stride0 as i32, src1, stride1 as i32);
    }
}
