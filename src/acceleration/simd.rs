pub(crate) trait SIMD: Copy + One + Zero + MulAdd<Output=Self> + AddAssign + MulAssign
+ Add<Output=Self> + Mul<Output=Self> {
    const LANES: usize;
    type SimdVec: Copy;

    unsafe fn simd_from(vals: &[Self]) -> Self::SimdVec;

    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec;

    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec);

    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_dup(val: Self) -> Self::SimdVec;

    /// Horizontal sum
    unsafe fn simd_sum(val: Self::SimdVec) -> Self;

    /// Horizontal multiply
    unsafe fn simd_prod(val: Self::SimdVec) -> Self;
}

use num::traits::MulAdd;
use num::{One, Zero};
#[cfg(neon_simd)]
use std::arch::aarch64::*;
use std::hint::assert_unchecked;
use std::ops::{Add, AddAssign, Mul, MulAssign};

#[cfg(neon_simd)]
impl SIMD for f32 {
    const LANES: usize = 4;
    type SimdVec = float32x4_t;

    unsafe fn simd_from(vals: &[Self]) -> Self::SimdVec {
        assert_unchecked(vals.len() == Self::LANES);
        
        let mut vec = Self::simd_dup(Self::default());
        vec = vsetq_lane_f32::<0>(vals[0], vec);
        vec = vsetq_lane_f32::<1>(vals[1], vec);
        vec = vsetq_lane_f32::<2>(vals[2], vec);
        vec = vsetq_lane_f32::<3>(vals[3], vec);
        
        vec
    }

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

    unsafe fn simd_prod(val: Self::SimdVec) -> Self {
        let tmp = vmul_f32(vget_low_f32(val), vget_high_f32(val));
        vget_lane_f32::<0>(tmp) * vget_lane_f32::<1>(tmp)
    }
}

#[cfg(neon_simd)]
impl SIMD for f64 {
    const LANES: usize = 2;
    type SimdVec = float64x2_t;

    unsafe fn simd_from(vals: &[Self]) -> Self::SimdVec {
        assert_unchecked(vals.len() == Self::LANES);

        let mut vec = Self::simd_dup(Self::default());
        vec = vsetq_lane_f64::<0>(vals[0], vec);
        vec = vsetq_lane_f64::<1>(vals[1], vec);

        vec
    }

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

    unsafe fn simd_prod(val: Self::SimdVec) -> Self {
        let tmp = vmul_f64(vget_low_f64(val), vget_high_f64(val));
        vget_lane_f64::<0>(tmp)
    }
}
