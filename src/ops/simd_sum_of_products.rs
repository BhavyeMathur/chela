use crate::acceleration::simd::SIMD;

pub(crate) trait SIMDSumOfProducts: SIMD {
    /// Performs a vectorized sum-of-products operation using fused multiply-add (FMA) instructions. 
    /// The operation performed is `*dst += scalar * (*src)` for `count` elements 
    /// beginning at `src` and `dst` and ending at `src + count` and `dst + count`.
    ///
    /// # Safety
    /// - `src` and `dst` must represent a valid array of `count` elements.
    /// - The arrays pointed to by `src` and `dst` do not overlap.
    #[cfg(neon_simd)]
    #[allow(clippy::erasing_op)]
    #[allow(clippy::identity_op)]
    unsafe fn simd_sum_of_products_muladd(scalar: Self,
                                          mut src: *const Self,
                                          mut dst: *mut Self,
                                          mut count: usize) {
        let scalarx = Self::simd_dup(scalar);

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(src.add(0 * Self::LANES));
            let b = Self::simd_load(src.add(1 * Self::LANES));
            let c = Self::simd_load(src.add(2 * Self::LANES));
            let d = Self::simd_load(src.add(3 * Self::LANES));

            let a_dst = Self::simd_load(dst.add(0 * Self::LANES));
            let b_dst = Self::simd_load(dst.add(1 * Self::LANES));
            let c_dst = Self::simd_load(dst.add(2 * Self::LANES));
            let d_dst = Self::simd_load(dst.add(3 * Self::LANES));

            let a_out = Self::simd_muladd(a_dst, scalarx, a);
            let b_out = Self::simd_muladd(b_dst, scalarx, b);
            let c_out = Self::simd_muladd(c_dst, scalarx, c);
            let d_out = Self::simd_muladd(d_dst, scalarx, d);

            Self::simd_store(dst.add(0 * Self::LANES), a_out);
            Self::simd_store(dst.add(1 * Self::LANES), b_out);
            Self::simd_store(dst.add(2 * Self::LANES), c_out);
            Self::simd_store(dst.add(3 * Self::LANES), d_out);

            count -= 4 * Self::LANES;
            dst = dst.add(4 * Self::LANES);
            src = src.add(4 * Self::LANES);
        }

        while count >= Self::LANES {
            let a = Self::simd_load(src);
            let a_dst = Self::simd_load(dst);
            let a_out = Self::simd_muladd(a_dst, scalarx, a);

            Self::simd_store(dst, a_out);

            count -= Self::LANES;
            dst = dst.add(Self::LANES);
            src = src.add(Self::LANES);
        }

        for _ in 0..count {
            *dst = scalar.mul_add(*src, *dst);
            dst = dst.add(1);
            src = src.add(1);
        }
    }

    /// Performs a vectorized sum-of-products operation using fused multiply-add (FMA).
    /// The operation performed is `*dst += sum(scalar * (*src))` for `count` elements
    /// beginning at `src` and `dst` and ending at `src + count` and `dst`.
    ///
    /// # Safety
    /// - `src` must represent a valid array of `count` elements.
    /// - `dst` must be a valid pointer to a scalar output.
    #[cfg(neon_simd)]
    #[allow(clippy::erasing_op)]
    #[allow(clippy::identity_op)]
    unsafe fn simd_sum_of_scaled_array(scalar: Self,
                                       mut src: *const Self,
                                       dst: *mut Self,
                                       mut count: usize) {
        let mut sum = Self::simd_dup(Self::zero());

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(src.add(0 * Self::LANES));
            let b = Self::simd_load(src.add(1 * Self::LANES));
            let c = Self::simd_load(src.add(2 * Self::LANES));
            let d = Self::simd_load(src.add(3 * Self::LANES));

            let ab = Self::simd_add(a, b);
            let cd = Self::simd_add(c, d);
            sum = Self::simd_add(sum, Self::simd_add(ab, cd));

            count -= 4 * Self::LANES;
            src = src.add(4 * Self::LANES);
        }

        while count >= Self::LANES {
            let a = Self::simd_load(src);
            sum = Self::simd_add(sum, a);

            count -= Self::LANES;
            src = src.add(Self::LANES);
        }

        let mut sum = Self::simd_sum(sum);
        for i in 0..count {
            sum += *src.add(i);
        }

        *dst = scalar.mul_add(sum, *dst);
    }

    /// Performs a dot product between `count` elements beginning at `src0` and `src1`
    /// and writing the result to `dst`
    ///
    /// # Safety
    /// - `src0` and `src1` must represent a valid array of `count` elements.
    /// - `dst` must be a valid pointer
    #[cfg(neon_simd)]
    #[allow(clippy::erasing_op)]
    #[allow(clippy::identity_op)]
    unsafe fn simd_dot_product(mut src0: *const Self,
                               mut src1: *const Self,
                               dst: *mut Self,
                               mut count: usize) {
        let mut sum = Self::simd_dup(Self::zero());

        while count >= 4 * Self::LANES {
            let a0 = Self::simd_load(src0.add(0 * Self::LANES));
            let b0 = Self::simd_load(src1.add(0 * Self::LANES));

            let a1 = Self::simd_load(src0.add(1 * Self::LANES));
            let b1 = Self::simd_load(src1.add(1 * Self::LANES));

            let a2 = Self::simd_load(src0.add(2 * Self::LANES));
            let b2 = Self::simd_load(src1.add(2 * Self::LANES));

            let a3 = Self::simd_load(src0.add(3 * Self::LANES));
            let b3 = Self::simd_load(src1.add(3 * Self::LANES));

            let ab0 = Self::simd_mul(a0, b0);
            let ab1 = Self::simd_mul(a1, b1);
            let ab2 = Self::simd_mul(a2, b2);
            let ab3 = Self::simd_mul(a3, b3);

            let ab01 = Self::simd_add(ab0, ab1);
            let ab23 = Self::simd_add(ab2, ab3);
            let ab0123 = Self::simd_add(ab01, ab23);

            sum = Self::simd_add(sum, ab0123);

            count -= 4 * Self::LANES;
            src0 = src0.add(4 * Self::LANES);
            src1 = src1.add(4 * Self::LANES);
        }

        while count >= Self::LANES {
            let a = Self::simd_load(src0);
            let b = Self::simd_load(src1);
            sum = Self::simd_muladd(sum, a, b);

            count -= Self::LANES;
            src0 = src0.add(Self::LANES);
            src1 = src1.add(Self::LANES);
        }

        let mut sum = Self::simd_sum(sum);
        for _ in 0..count {
            sum = (*src0).mul_add(*src1, sum);
            src0 = src0.add(1);
            src1 = src1.add(1);
        }

        *dst += sum;
    }
}

impl<T: SIMD> SIMDSumOfProducts for T {}
