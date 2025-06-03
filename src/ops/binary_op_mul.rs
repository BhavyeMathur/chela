use crate::impl_default_binary_op_trait;
use crate::define_binary_op_trait;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use paste::paste;
use std::ops::Mul;

define_binary_op_trait!(BinaryOpMul, Mul, mul, *;
                        i8, i16, i32, i64, i128, isize,
                        u8, u16, u32, u64, u128, usize);

impl BinaryOpMul for f32 {
    #[cfg(apple_vdsp)]
    unsafe fn mul_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsmul;
        vDSP_vsmul(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(neon_simd)]
    unsafe fn mul_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_mul_stride_1_1(lhs, rhs, dst, count);
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn mul_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vmul;
        vDSP_vmul(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}

impl BinaryOpMul for f64 {
    #[cfg(apple_vdsp)]
    unsafe fn mul_stride_n_0(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vsmulD;
        vDSP_vsmulD(lhs, lhs_stride as isize, rhs, dst, 1, count);
    }

    #[cfg(neon_simd)]
    unsafe fn mul_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_mul_stride_1_1(lhs, rhs, dst, count);
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn mul_stride_n_n(lhs: *const Self, lhs_stride: usize,
                             rhs: *const Self, rhs_stride: usize,
                             dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vmulD;
        vDSP_vmulD(lhs, lhs_stride as isize, rhs, rhs_stride as isize, dst, 1, count);
    }
}
