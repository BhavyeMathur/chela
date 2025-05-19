use crate::dtype::{IntegerDataType, NumericDataType};
use crate::tensor::MAX_DIMS;
use std::hint::assert_unchecked;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

pub(super) fn get_sum_of_products_function<const N: usize, T: EinsumDataType>(strides: &[usize; N])
                                                                              -> unsafe fn(ptrs: &[*mut T; N], stride: &[usize; N], count: usize) {
    if N == 3 { // 2 operands + 1 output
        let mut code = if strides[0] == 0 { 0 } else { if strides[0] == 1 { 4 } else { 8 } };
        code += if strides[1] == 0 { 0 } else { if strides[1] == 1 { 2 } else { 8 } };
        code += if strides[2] == 0 { 0 } else { if strides[2] == 1 { 1 } else { 8 } };

        match code {
            2 => { return <T as EinsumDataType>::operand_strides_0_1_out_stride_0; }
            3 => { return <T as EinsumDataType>::operand_strides_0_1_out_stride_1; }
            _ => {}
        }
    }

    <T as EinsumDataType>::generic
}

pub(super) trait EinsumDataType: NumericDataType {
    #[inline(always)]
    unsafe fn generic<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(count > 0);
        assert_unchecked(N > 0 && N <= MAX_DIMS);

        let dst = ptrs[N - 1];
        let ptrs = &ptrs[0..N - 1];

        let mut k = count;
        while k != 0 {
            k -= 1;

            let dst = dst.add(k * strides[N - 1]);
            *dst += ptrs.iter().zip(strides.iter())
                        .map(|(ptr, stride)| *ptr.add(k * stride))
                        .product();
        }
    }

    #[inline(always)]
    unsafe fn operand_strides_0_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], _: &[usize; N], count: usize) {
        assert_unchecked(count > 0);

        let dst = ptrs[N - 1];
        let ptrs = &ptrs[0..N - 1];

        let value0 = *ptrs[0];
        let data1 = ptrs[1];

        let mut sum = Self::zero();
        for i in 0..count {
            sum += *data1.add(i);
        }

        *dst = value0.mul_add(sum, *dst);
    }

    #[inline(always)]
    unsafe fn operand_strides_0_1_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], _: &[usize; N], count: usize) {
        assert_unchecked(count > 0);

        let mut dst = ptrs[N - 1];
        let ptrs = &ptrs[0..N - 1];

        let value0 = *ptrs[0];
        let mut data1 = ptrs[1];

        for _ in 0..count {
            *dst = value0.mul_add(*data1, *dst);
            dst = dst.add(1);
            data1 = data1.add(1);
        }
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
    unsafe fn operand_strides_0_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], _: &[usize; N], mut count: usize) {
        assert_unchecked(count > 0);

        let dst = ptrs[N - 1];
        let value0 = *ptrs[0];
        let mut data1 = ptrs[1];

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
    unsafe fn operand_strides_0_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], _: &[usize; N], mut count: usize) {
        assert_unchecked(count > 0);

        let dst = ptrs[N - 1];
        let value0 = *ptrs[0];
        let mut data1 = ptrs[1];

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

    unsafe fn operand_strides_0_1_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], _: &[usize; N], mut count: usize) {
        assert_unchecked(count > 0);

        let mut dst = ptrs[N - 1];
        let value0 = *ptrs[0];
        let value0x4 = vdupq_n_f32(value0);
        let mut data1 = ptrs[1];

        while count >= 16 {
            let a = vld1q_f32(data1);
            let b = vld1q_f32(data1.add(4));
            let c = vld1q_f32(data1.add(8));
            let d = vld1q_f32(data1.add(12));

            let a_dst = vld1q_f32(dst);
            let b_dst = vld1q_f32(dst.add(4));
            let c_dst = vld1q_f32(dst.add(8));
            let d_dst = vld1q_f32(dst.add(12));

            let a_out = vfmaq_f32(a_dst, value0x4, a);
            let b_out = vfmaq_f32(b_dst, value0x4, b);
            let c_out = vfmaq_f32(c_dst, value0x4, c);
            let d_out = vfmaq_f32(d_dst, value0x4, d);

            vst1q_f32(dst, a_out);
            vst1q_f32(dst.add(4), b_out);
            vst1q_f32(dst.add(8), c_out);
            vst1q_f32(dst.add(12), d_out);

            dst = dst.add(16);
            data1 = data1.add(16);
            count -= 16;
        }

        while count >= 4 {
            let a = vld1q_f32(data1);
            let a_dst = vld1q_f32(dst);
            let a_out = vfmaq_f32(a_dst, value0x4, a);

            vst1q_f32(dst, a_out);

            data1 = data1.add(4);
            count -= 4;
        }

        for _ in 0..count {
            *dst = value0.mul_add(*data1, *dst);
            dst = dst.add(1);
            data1 = data1.add(1);
        }
    }
}
