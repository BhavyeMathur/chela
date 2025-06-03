use crate::define_binary_op_trait;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use crate::{impl_default_trait_for_dtypes, simd_binary_op_specializations};
use paste::paste;
use std::ops::Add;
use std::ptr::addr_of;


define_binary_op_trait!(BinaryOpAdd, Add, add, +;
                        i8, i16, i64, i128, isize,
                        u8, u16, u64, u128, usize);

impl BinaryOpAdd for f32 {
    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn add_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsadd;
        vDSP_vsadd(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn add_stride_0_n(lhs: *const Self,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsadd;
        vDSP_vsadd(rhs, rhs_stride as isize, lhs, dst, 1, count);
    }

    simd_binary_op_specializations!(add);

    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn add_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vadd;
        vDSP_vadd(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpAdd for f64 {
    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn add_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddD;
        vDSP_vsaddD(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(all(apple_vdsp, not(neon_simd)))]
    unsafe fn add_stride_0_n(lhs: *const Self,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddD;
        vDSP_vsaddD(rhs, rhs_stride as isize, lhs, dst, 1, count);
    }

    simd_binary_op_specializations!(add);

    #[cfg(all(apple_vdsp, not(neon_simd)))]
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
    unsafe fn add_stride_0_n(lhs: *const Self,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddi;
        vDSP_vsaddi(rhs, rhs_stride as isize, lhs, dst, 1, count);
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
    unsafe fn add_stride_0_n(lhs: *const Self,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsaddi;
        vDSP_vsaddi(rhs as *const i32, rhs_stride as isize,
                    lhs as *const i32,
                    dst as *mut i32, 1, count);
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
