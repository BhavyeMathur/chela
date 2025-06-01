pub(crate) trait SIMD: Copy + Zero + AddAssign {
    const LANES: usize;
    type SimdVec;

    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec;

    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec);

    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_dup(val: Self) -> Self::SimdVec;

    unsafe fn simd_sum(val: Self::SimdVec) -> Self;

    unsafe fn simd_sum_contiguous(mut ptr: *const Self, mut count: usize) -> Self {
        let mut output = Self::zero();

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(ptr.add(0 * Self::LANES));
            let b = Self::simd_load(ptr.add(1 * Self::LANES));
            let c = Self::simd_load(ptr.add(2 * Self::LANES));
            let d = Self::simd_load(ptr.add(3 * Self::LANES));

            let ab = Self::simd_add(a, b);
            let cd = Self::simd_add(c, d);
            let abcd = Self::simd_add(ab, cd);

            count -= 4 * Self::LANES;
            ptr = ptr.add(4 * Self::LANES);

            output += Self::simd_sum(abcd);
        }

        for _ in 0..count {
            output += *ptr;
            ptr = ptr.add(1);
        }

        output
    }
}

#[cfg(neon_simd)]
use std::arch::aarch64::*;
use std::ops::AddAssign;
use num::Zero;

#[cfg(neon_simd)]
impl SIMD for f32 {
    const LANES: usize = 4;
    type SimdVec = float32x4_t;

    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec {
        vld1q_f32(ptr)
    }

    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec) {
        vst1q_f32(ptr, val)
    }

    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vaddq_f32(lhs, rhs)
    }

    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vmulq_f32(lhs, rhs)
    }

    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vfmaq_f32(sum, lhs, rhs)
    }

    unsafe fn simd_dup(val: Self) -> Self::SimdVec {
        vdupq_n_f32(val)
    }

    unsafe fn simd_sum(val: Self::SimdVec) -> Self {
        vaddvq_f32(val)
    }
}

#[cfg(neon_simd)]
impl SIMD for f64 {
    const LANES: usize = 2;
    type SimdVec = float64x2_t;

    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec {
        vld1q_f64(ptr)
    }

    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec) {
        vst1q_f64(ptr, val)
    }

    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vaddq_f64(lhs, rhs)
    }

    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vmulq_f64(lhs, rhs)
    }

    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vfmaq_f64(sum, lhs, rhs)
    }

    unsafe fn simd_dup(val: Self) -> Self::SimdVec {
        vdupq_n_f64(val)
    }

    unsafe fn simd_sum(val: Self::SimdVec) -> Self {
        vaddvq_f64(val)
    }
}
