use crate::impl_default_binary_op_trait;
use crate::define_binary_op_trait;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use paste::paste;
use std::ops::Add;

define_binary_op_trait!(BinaryOpAdd, Add, add, +;
                        i8, i16, i64, i128, isize,
                        u8, u16, u64, u128, usize);

impl BinaryOpAdd for f32 {
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsadd;
        vDSP_vsadd(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_add_stride_1_1(lhs, rhs, dst, count);
    }

    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vadd;
        vDSP_vadd(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpAdd for f64 {
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddD;
        vDSP_vsaddD(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_add_stride_1_1(lhs, rhs, dst, count);
    }

    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vaddD;
        vDSP_vaddD(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpAdd for i32 {
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddi;
        vDSP_vsaddi(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vaddi;
        vDSP_vaddi(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpAdd for u32 {
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddi;
        vDSP_vsaddi(lhs as *const i32, lhs_stride as isize,
                    rhs as *const i32, dst as *mut i32, 1, count);
    }

    #[cfg(apple_vdsp)]
    unsafe fn add_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vaddi;
        vDSP_vaddi(lhs as *const i32, lhs_stride as isize,
                   rhs as *const i32, rhs_stride as isize,
                   dst as *mut i32, 1, count);
    }
}
