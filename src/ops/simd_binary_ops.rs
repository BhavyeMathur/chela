use crate::acceleration::simd::Simd;

pub(crate) trait SimdBinaryOps: Simd {
    #[cfg(neon_simd)]
    unsafe fn simd_add_stride_1_1(mut lhs: *const Self, mut rhs: *const Self, mut dst: *mut Self, mut count: usize) {
        while count >= 4 * Self::LANES {
            let a0 = Self::simd_load(lhs.add(0 * Self::LANES));
            let b0 = Self::simd_load(rhs.add(0 * Self::LANES));

            let a1 = Self::simd_load(lhs.add(1 * Self::LANES));
            let b1 = Self::simd_load(rhs.add(1 * Self::LANES));

            let a2 = Self::simd_load(lhs.add(2 * Self::LANES));
            let b2 = Self::simd_load(rhs.add(2 * Self::LANES));

            let a3 = Self::simd_load(lhs.add(3 * Self::LANES));
            let b3 = Self::simd_load(rhs.add(3 * Self::LANES));

            let ab0 = Self::simd_add(a0, b0);
            let ab1 = Self::simd_add(a1, b1);
            let ab2 = Self::simd_add(a2, b2);
            let ab3 = Self::simd_add(a3, b3);

            Self::simd_store(dst.add(0 * Self::LANES), ab0);
            Self::simd_store(dst.add(1 * Self::LANES), ab1);
            Self::simd_store(dst.add(2 * Self::LANES), ab2);
            Self::simd_store(dst.add(3 * Self::LANES), ab3);

            count -= 4 * Self::LANES;
            lhs = lhs.add(4 * Self::LANES);
            rhs = rhs.add(4 * Self::LANES);
            dst = dst.add(4 * Self::LANES);
        }

        while count != 0 {
            *dst = *lhs + *rhs;

            count -= 1;
            lhs = lhs.add(1);
            rhs = rhs.add(1);
            dst = dst.add(1);
        }
    }
}

impl<T: Simd> SimdBinaryOps for T {}
