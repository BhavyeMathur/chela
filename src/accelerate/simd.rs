pub(crate) trait SIMD {
    const LANES: usize;
    type SimdVec;

    unsafe fn simd_load(ptr: *const Self) -> Self::SimdVec;

    unsafe fn simd_store(ptr: *mut Self, val: Self::SimdVec);

    unsafe fn simd_add(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_mul(lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_muladd(sum: Self::SimdVec, lhs: Self::SimdVec, rhs: Self::SimdVec) -> Self::SimdVec;

    unsafe fn simd_dup(val: Self) -> Self::SimdVec;

    unsafe fn simd_sum(val: Self::SimdVec) -> Self;
}

#[cfg(neon_simd)]
use std::arch::aarch64::*;

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
