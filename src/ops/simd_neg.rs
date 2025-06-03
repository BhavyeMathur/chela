use crate::acceleration::simd::Simd;
use std::ops::Neg;

pub(crate) trait SimdNeg: Simd + Neg<Output=Self> {
    #[cfg(neon_simd)]
    unsafe fn simd_neg_stride_1(mut src: *const Self, mut dst: *mut Self, mut count: usize) {
        while count >= 4 * Self::LANES {
            let a0 = Self::simd_load(src.add(0 * Self::LANES));
            let a1 = Self::simd_load(src.add(1 * Self::LANES));
            let a2 = Self::simd_load(src.add(2 * Self::LANES));
            let a3 = Self::simd_load(src.add(3 * Self::LANES));

            let b0 = Self::simd_neg(a0);
            let b1 = Self::simd_neg(a1);
            let b2 = Self::simd_neg(a2);
            let b3 = Self::simd_neg(a3);

            Self::simd_store(dst.add(0 * Self::LANES), b0);
            Self::simd_store(dst.add(1 * Self::LANES), b1);
            Self::simd_store(dst.add(2 * Self::LANES), b2);
            Self::simd_store(dst.add(3 * Self::LANES), b3);

            count -= 4 * Self::LANES;
            src = src.add(4 * Self::LANES);
            dst = dst.add(4 * Self::LANES);
        }

        while count != 0 {
            *dst = -*src;

            count -= 1;
            src = src.add(1);
            dst = dst.add(1);
        }
    }

    #[cfg(neon_simd)]
    unsafe fn simd_neg_stride_n(mut src: *const Self, stride: usize, mut dst: *mut Self, mut count: usize) {
        while count >= 4 * Self::LANES {
            let a0 = Self::simd_vec_from_stride(src.add(0 * stride * Self::LANES), stride);
            let a1 = Self::simd_vec_from_stride(src.add(1 * stride * Self::LANES), stride);
            let a2 = Self::simd_vec_from_stride(src.add(2 * stride * Self::LANES), stride);
            let a3 = Self::simd_vec_from_stride(src.add(3 * stride * Self::LANES), stride);

            let b0 = Self::simd_neg(a0);
            let b1 = Self::simd_neg(a1);
            let b2 = Self::simd_neg(a2);
            let b3 = Self::simd_neg(a3);

            Self::simd_store(dst.add(0 * Self::LANES), b0);
            Self::simd_store(dst.add(1 * Self::LANES), b1);
            Self::simd_store(dst.add(2 * Self::LANES), b2);
            Self::simd_store(dst.add(3 * Self::LANES), b3);

            count -= 4 * Self::LANES;
            src = src.add(4 * stride * Self::LANES);
            dst = dst.add(4 * Self::LANES);
        }

        while count != 0 {
            *dst = -*src;

            count -= 1;
            src = src.add(stride);
            dst = dst.add(1);
        }
    }
}

impl<T: Simd + Neg<Output=Self>> SimdNeg for T {}
