use crate::dtype::{IntegerDataType, NumericDataType};
use crate::tensor::MAX_DIMS;
use std::hint::assert_unchecked;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

pub(super) fn get_sum_of_products_function<const N: usize, T: EinsumDataType>(strides: [usize; N],
                                                                              output_stride: usize)
                                                                              -> unsafe fn(&[*const T; N], &[usize; N], usize, usize, *mut T) {
    if N == 2 {
        let mut code = if strides[0] == 0 { 0 } else { if strides[0] == 1 { 4 } else { 8 } };
        code += if strides[1] == 0 { 0 } else { if strides[1] == 1 { 2 } else { 8 } };
        code += if output_stride == 0 { 0 } else { if output_stride == 1 { 1 } else { 8 } };

        if code == 2 {
            return <T as EinsumDataType>::stride0_contig_outstride0_two;
        }
    }

    <T as EinsumDataType>::generic
}

pub(super) trait EinsumDataType: NumericDataType {
    #[inline(always)]
    unsafe fn generic<const N: usize>(ptrs: &[*const Self; N], strides: &[usize; N], output_stride: usize, count: usize, dst: *mut Self) {
        assert_unchecked(count > 0);
        assert_unchecked(N > 0 && N <= MAX_DIMS);

        let mut k = count;
        while k != 0 {
            k -= 1;

            let dst = dst.add(k * output_stride);
            *dst += ptrs.iter().zip(strides.iter())
                        .map(|(ptr, stride)| *ptr.add(k * stride))
                        .product();
        }
    }

    #[inline(always)]
    unsafe fn stride0_contig_outstride0_two<const N: usize>(ptrs: &[*const Self; N], _: &[usize; N], _: usize, count: usize, dst: *mut Self) {
        assert_unchecked(count > 0);

        let value0 = *ptrs[0];
        let data1 = ptrs[1];

        let mut sum = Self::zero();
        for i in 0..count {
            sum += *data1.add(i);
        }

        *dst = value0.mul_add(sum, *dst);
    }
}

impl<T: IntegerDataType> EinsumDataType for T {}

#[cfg(not(target_arch = "aarch64"))]
impl EinsumDataType for f32 {}

#[cfg(not(target_arch = "aarch64"))]
impl EinsumDataType for f64 {}

#[cfg(target_arch = "aarch64")]
impl EinsumDataType for f64 {
    #[inline(always)]
    unsafe fn stride0_contig_outstride0_two<const N: usize>(ptrs: &[*const Self; N], _: &[usize; N], _: usize, count: usize, dst: *mut Self) {
        assert_unchecked(count > 0);

        let value0 = *ptrs[0];
        let mut data1 = ptrs[1];
        let mut count = count as isize;

        let mut sum = vdupq_n_f64(0.0);

        while count >= 8 {
            let a = vld1q_f64(data1);
            let b = vld1q_f64(data1.add(8));

            let ab = vaddq_f64(a, b);
            sum = vaddq_f64(sum, ab);

            data1 = data1.add(8);
            count -= 8;
        }

        let sum_array: [f64; 2] = core::mem::transmute(sum);
        let mut sum = sum_array.iter().copied().sum::<f64>();

        for i in 0..count {
            sum += *data1.add(i as usize);
        }

        *dst = value0.mul_add(sum, *dst);
    }
}

#[cfg(target_arch = "aarch64")]
impl EinsumDataType for f32 {
    #[inline(always)]
    unsafe fn stride0_contig_outstride0_two<const N: usize>(ptrs: &[*const Self; N], _: &[usize; N], output_stride: usize, count: usize, dst: *mut Self) {
        assert_unchecked(count > 0);

        let value0 = *ptrs[0];
        let mut data1 = ptrs[1];
        let mut count = count as isize;

        let mut sum = vdupq_n_f32(0.0);

        while count >= 16 {
            let a = vld1q_f32(data1);
            let b = vld1q_f32(data1.add(4));
            let c = vld1q_f32(data1.add(8));
            let d = vld1q_f32(data1.add(12));

            let ab = vaddq_f32(a, b);
            let cd = vaddq_f32(c, d);
            sum = vaddq_f32(sum, vaddq_f32(ab, cd));

            data1 = data1.add(16);
            count -= 16;
        }

        let sum_array: [f32; 4] = core::mem::transmute(sum);
        let mut sum = sum_array.iter().copied().sum::<f32>();

        for i in 0..count {
            sum += *data1.add(i as usize);
        }

        *dst = value0.mul_add(sum, *dst);
    }
}
