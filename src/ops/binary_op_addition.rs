use crate::ndarray::collapse_contiguous::collapse_to_uniform_stride;
use std::ops::Add;
use crate::flat_index_generator::FlatIndexGenerator;
use crate::{IntegerDataType};
use crate::ops::fill::Fill;

pub(crate) trait BinaryOpAdd: Add<Output=Self> + Sized + Copy {
    unsafe fn add_stride_1_1(mut lhs: *const Self, mut rhs: *const Self, mut dst: *mut Self, mut count: usize) {
        while count != 0 {
            *dst = *lhs + *rhs;

            count -= 1;
            lhs = lhs.add(1);
            rhs = rhs.add(1);
            dst = dst.add(1);
        }
    }
    
    unsafe fn add_unspecialized(lhs: *const Self, lhs_shape: &[usize], lhs_stride: &[usize],
                                rhs: *const Self, rhs_shape: &[usize], rhs_stride: &[usize],
                                mut dst: *mut Self) {
        let lhs_indices = FlatIndexGenerator::from(lhs_shape, lhs_stride);
        let rhs_indices = FlatIndexGenerator::from(rhs_shape, rhs_stride);

        for (lhs_index, rhs_index) in lhs_indices.zip(rhs_indices) {
            *dst = *lhs.add(lhs_index) + *rhs.add(rhs_index);
            dst = dst.add(1);
        }
    }
    
    unsafe fn add(lhs: *const Self, lhs_stride: &[usize],
                  rhs: *const Self, rhs_stride: &[usize],
                  dst: *mut Self, shape: &[usize]) {
        let (lhs_shape, lhs_stride) = collapse_to_uniform_stride(shape, &lhs_stride);
        let (rhs_shape, rhs_stride) = collapse_to_uniform_stride(shape, &rhs_stride);

        let lhs_dims = lhs_shape.len();
        let rhs_dims = rhs_shape.len();
        
        let lhs_inner_stride = lhs_stride[lhs_dims - 1];
        let rhs_inner_stride = rhs_stride[rhs_dims - 1];
        
        if lhs_dims == 1 && rhs_dims == 1 { // both operands have a uniform stride
            if lhs_inner_stride == 1 && rhs_inner_stride == 1 {  // both operands are contiguous
                return Self::add_stride_1_1(lhs, rhs, dst, lhs_shape[0]);
            }
        }

        Self::add_unspecialized(lhs, &lhs_shape, &lhs_stride,
                                rhs, &rhs_shape, &rhs_stride,
                                dst);
    }
}

impl BinaryOpAdd for i8 {}
impl BinaryOpAdd for i16 {}

impl BinaryOpAdd for i64 {}
impl BinaryOpAdd for i128 {}
impl BinaryOpAdd for isize {}

impl BinaryOpAdd for u8 {}
impl BinaryOpAdd for u16 {}
impl BinaryOpAdd for u64 {}
impl BinaryOpAdd for u128 {}
impl BinaryOpAdd for usize {}

impl BinaryOpAdd for f32 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_add_stride_1_1(lhs, rhs, dst, count);
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vadd;
        vDSP_vadd(lhs, 1, rhs, 1, dst, 1, count);
    }
}

impl BinaryOpAdd for f64 {
    #[cfg(all(neon_simd, not(apple_vdsp)))]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::ops::simd_binary_ops::SimdBinaryOps;
        Self::simd_add_stride_1_1(lhs, rhs, dst, count);
    }
    
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vaddD;
        vDSP_vaddD(lhs, 1, rhs, 1, dst, 1, count);
    }
}

impl BinaryOpAdd for i32 {
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vaddi;
        vDSP_vaddi(lhs, 1, rhs, 1, dst, 1, count);
    }
}

impl BinaryOpAdd for u32 {
    #[cfg(apple_vdsp)]
    unsafe fn add_stride_1_1(lhs: *const Self, rhs: *const Self, dst: *mut Self, count: usize) {
        use crate::acceleration::vdsp::vDSP_vaddi;
        vDSP_vaddi(lhs as *const i32, 1, rhs as *const i32, 1, dst as *mut i32, 1, count);
    }
}
