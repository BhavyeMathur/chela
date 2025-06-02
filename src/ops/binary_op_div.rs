use crate::define_binary_op_trait;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use paste::paste;
use std::ops::Div;


define_binary_op_trait!(BinaryOpDiv, Div, div, /);

impl BinaryOpDiv for f32 {
    // #[cfg(apple_vdsp)]
    // unsafe fn div_stride_n_0(lhs: *const Self, lhs_stride: usize,
    //                          rhs: *const Self, dst: *mut Self, count: usize) {
    //     use crate::acceleration::vdsp::vDSP_vsdiv;
    //     vDSP_vsdiv(lhs, lhs_stride as isize, rhs, dst, 1, count);
    // }
    //
    // #[cfg(all(neon_simd, not(apple_vdsp)))]
    // unsafe fn div_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
    //     use crate::ops::simd_binary_ops::SimdBinaryOps;
    //     Self::simd_div_stride_1_1(lhs, rhs, dst, count);
    // }
    //
    // #[cfg(apple_vdsp)]
    // unsafe fn div_stride_n_n(lhs: *const Self, lhs_stride: usize,
    //                          rhs: *const Self, rhs_stride: usize,
    //                          dst: *mut Self, count: usize) {
    //     use crate::acceleration::vdsp::vDSP_vdiv;
    //     vDSP_vdiv(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    // }
}

impl BinaryOpDiv for f64 {
    // #[cfg(apple_vdsp)]
    // unsafe fn div_stride_n_0(lhs: *const Self, lhs_stride: usize,
    //                          rhs: *const Self, dst: *mut Self, count: usize) {
    //     use crate::acceleration::vdsp::vDSP_vsdivD;
    //     vDSP_vsdivD(lhs, lhs_stride as isize, rhs, dst, 1, count);
    // }
    //
    // #[cfg(all(neon_simd, not(apple_vdsp)))]
    // unsafe fn div_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
    //     use crate::ops::simd_binary_ops::SimdBinaryOps;
    //     Self::simd_div_stride_1_1(lhs, rhs, dst, count);
    // }
    //
    // #[cfg(apple_vdsp)]
    // unsafe fn div_stride_n_n(lhs: *const Self, lhs_stride: usize,
    //                          rhs: *const Self, rhs_stride: usize,
    //                          dst: *mut Self, count: usize) {
    //     use crate::acceleration::vdsp::vDSP_vdivD;
    //     vDSP_vdivD(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    // }
}
