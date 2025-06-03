pub(crate) trait Simd: Copy + One + Zero + Bounded + PartialOrd
+ MulAdd<Output=Self> + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self>
+ AddAssign + MulAssign
{
    const LANES: usize;
    type SimdVec: Copy;

    unsafe fn simd_vec_from_stride(ptr: *const Self, stride: usize) -> Self::SimdVec {
        if Self::LANES == 4 {
            let a = *ptr.add(0 * stride);
            let b = *ptr.add(1 * stride);
            let c = *ptr.add(2 * stride);
            let d = *ptr.add(3 * stride);

            Self::simd_from_array(&[a, b, c, d])
        } else if Self::LANES == 2 {
            let a = *ptr.add(0 * stride);
            let b = *ptr.add(1 * stride);

            Self::simd_from_array(&[a, b])
        } else if Self::LANES == 1 {
            Self::simd_from_array(&[*ptr])
        } else {
            panic!("unimplemented SIMD sum uniform")
        }
    }

    unsafe fn simd_from_array(_: &[Self]) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_from_constant(_: Self) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_load(_: *const Self) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_store(_: *mut Self, _: Self::SimdVec) {
        unimplemented!()
    }

    unsafe fn simd_neg(_: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_add(_: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_sub(_: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_mul(_: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_div(_: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_muladd(_: Self::SimdVec, _: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_min(_: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_max(_: Self::SimdVec, _: Self::SimdVec) -> Self::SimdVec {
        unimplemented!()
    }

    unsafe fn simd_horizontal_sum(_: Self::SimdVec) -> Self {
        unimplemented!()
    }

    unsafe fn simd_horizontal_mul(_: Self::SimdVec) -> Self {
        unimplemented!()
    }

    unsafe fn simd_horizontal_min(_: Self::SimdVec) -> Self {
        unimplemented!()
    }

    unsafe fn simd_horizontal_max(_: Self::SimdVec) -> Self {
        unimplemented!()
    }
}

use num::traits::MulAdd;
use num::{Bounded, One, Zero};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

#[cfg(neon_simd)]
use std::arch::aarch64::*;

#[cfg(neon_simd)]
use std::hint::assert_unchecked;

impl Simd for f32 {
    const LANES: usize = 4;

    #[cfg(neon_simd)]
    type SimdVec = float32x4_t;

    #[cfg(not(neon_simd))]
    type SimdVec = bool;

    #[cfg(neon_simd)]
    unsafe fn simd_from_array(vals: &[Self]) -> Self::SimdVec {
        assert_unchecked(vals.len() == Self::LANES);

        let mut vec = Self::simd_from_constant(Self::default());
        vec = vsetq_lane_f32::<0>(vals[0], vec);
        vec = vsetq_lane_f32::<1>(vals[1], vec);
        vec = vsetq_lane_f32::<2>(vals[2], vec);
        vec = vsetq_lane_f32::<3>(vals[3], vec);

        vec
    }

    #[cfg(neon_simd)]
    unsafe fn simd_from_constant(val: Self) -> Self::SimdVec {
        vdupq_n_f32(val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec {
        vld1q_f32(ptr)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec) {
        vst1q_f32(ptr, val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_neg(vec: Self::SimdVec) -> Self::SimdVec {
        vnegq_f32(vec)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vaddq_f32(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_sub(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vsubq_f32(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vmulq_f32(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_div(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vdivq_f32(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vfmaq_f32(sum, lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_min(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vminq_f32(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_max(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vmaxq_f32(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_sum(val: Self::SimdVec) -> Self {
        vaddvq_f32(val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_mul(val: Self::SimdVec) -> Self {
        let tmp = vmul_f32(vget_low_f32(val), vget_high_f32(val));
        vget_lane_f32::<0>(tmp) * vget_lane_f32::<1>(tmp)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_min(val: Self::SimdVec) -> Self {
        vminvq_f32(val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_max(val: Self::SimdVec) -> Self {
        vmaxvq_f32(val)
    }
}

impl Simd for f64 {
    const LANES: usize = 2;

    #[cfg(neon_simd)]
    type SimdVec = float64x2_t;

    #[cfg(not(neon_simd))]
    type SimdVec = bool;

    #[cfg(neon_simd)]
    unsafe fn simd_from_array(vals: &[Self]) -> Self::SimdVec {
        assert_unchecked(vals.len() == Self::LANES);

        let mut vec = Self::simd_from_constant(Self::default());
        vec = vsetq_lane_f64::<0>(vals[0], vec);
        vec = vsetq_lane_f64::<1>(vals[1], vec);

        vec
    }

    #[cfg(neon_simd)]
    unsafe fn simd_from_constant(val: Self) -> Self::SimdVec {
        vdupq_n_f64(val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec {
        vld1q_f64(ptr)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec) {
        vst1q_f64(ptr, val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_neg(vec: Self::SimdVec) -> Self::SimdVec {
        vnegq_f64(vec)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vaddq_f64(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_sub(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vsubq_f64(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vmulq_f64(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_div(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vdivq_f64(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vfmaq_f64(sum, lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_min(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vminq_f64(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_max(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec {
        vmaxq_f64(lhs, rhs)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_sum(val: Self::SimdVec) -> Self {
        vaddvq_f64(val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_mul(val: Self::SimdVec) -> Self {
        let tmp = vmul_f64(vget_low_f64(val), vget_high_f64(val));
        vget_lane_f64::<0>(tmp)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_min(val: Self::SimdVec) -> Self {
        vminvq_f64(val)
    }

    #[cfg(neon_simd)]
    unsafe fn simd_horizontal_max(val: Self::SimdVec) -> Self {
        vmaxvq_f64(val)
    }
}
