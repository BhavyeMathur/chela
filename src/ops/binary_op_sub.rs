use crate::define_binary_op_trait;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::impl_default_binary_op_trait;
use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use paste::paste;
use std::ops::Sub;

define_binary_op_trait!(BinaryOpSub, Sub, sub, -;
                        i8, i16, i32, i64, i128, isize,
                        u8, u16, u32, u64, u128, usize);

impl BinaryOpSub for f32 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sub_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_sub_stride_1_1(lhs, rhs, dst, count);
    }

    #[cfg(apple_vdsp)]
    unsafe fn sub_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsub;
        vDSP_vsub(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpSub for f64 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn sub_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_sub_stride_1_1(lhs, rhs, dst, count);
    }

    #[cfg(apple_vdsp)]
    unsafe fn sub_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsubD;
        vDSP_vsubD(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}
