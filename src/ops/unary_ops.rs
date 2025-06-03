use crate::flat_index_generator::FlatIndexGenerator;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::ops::simd_neg::SimdNeg;
use crate::FloatDataType;
use std::ops::Neg;


pub(crate) trait UnaryOps: Neg<Output=Self> + Sized + Copy {
    unsafe fn neg_stride_0(operand: *const Self, dst: *mut Self, count: usize) {
        Self::neg_stride_n(operand, 0, dst, count);
    }

    unsafe fn neg_stride_1(operand: *const Self, dst: *mut Self, count: usize) {
        Self::neg_stride_n(operand, 1, dst, count);
    }

    unsafe fn neg_stride_n(mut operand: *const Self, stride: usize, mut dst: *mut Self, mut count: usize) {
        while count != 0 {
            *dst = -*operand;

            count -= 1;
            operand = operand.add(stride);
            dst = dst.add(1);
        }
    }

    unsafe fn neg_unspecialized(operand: *const Self, shape: &[usize], stride: &[usize], mut dst: *mut Self) {
        let indices = FlatIndexGenerator::from(shape, stride);

        for index in indices {
            unsafe {
                *dst = -*operand.add(index);
                dst = dst.add(1);
            }
        }
    }

    unsafe fn neg(operand: *const Self, shape: &[usize], stride: &[usize], dst: *mut Self) {
        // special case for scalar tensor
        if shape.is_empty() {
            *dst = -*operand;
            return;
        }

        let (shape, stride) = collapse_to_uniform_stride(shape, stride);

        if shape.len() == 1 {
            if stride[0] == 0 {
                return Self::neg_stride_0(operand, dst, shape[0]);
            }

            if stride[0] == 1 {
                return Self::neg_stride_1(operand, dst, shape[0]);
            }

            return Self::neg_stride_n(operand, stride[0], dst, shape[0]);
        }

        Self::neg_unspecialized(operand, &shape, &stride, dst);
    }
}


impl_default_trait_for_dtypes!(UnaryOps, i8, i16, i32, i64, i128, isize);


impl<T: FloatDataType + SimdNeg> UnaryOps for T {
    #[cfg(neon_simd)]
    unsafe fn neg_stride_1(operand: *const Self, dst: *mut Self, count: usize) {
        Self::simd_neg_stride_1(operand, dst, count);
    }

    #[cfg(neon_simd)]
    unsafe fn neg_stride_n(operand: *const Self, stride: usize, dst: *mut Self, count: usize) {
        Self::simd_neg_stride_n(operand, stride, dst, count);
    }
}
