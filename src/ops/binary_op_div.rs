use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use crate::{define_binary_op_trait, simd_binary_op_specializations};
use paste::paste;
use std::ops::Div;
use std::ptr::addr_of;


define_binary_op_trait!(BinaryOpDiv, Div, div, /);

impl BinaryOpDiv for f32 {
    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn div_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsdiv;
        vDSP_vsdiv(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    simd_binary_op_specializations!(div);

    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn div_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vdiv;
        vDSP_vdiv(rhs, rhs_stride as isize, lhs, lhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpDiv for f64 {
    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn div_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsdivD;
        vDSP_vsdivD(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    simd_binary_op_specializations!(div);

    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn div_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vdivD;
        vDSP_vdivD(rhs, rhs_stride as isize, lhs, lhs_stride as isize, dst, 1, count);
    }
}
