#![allow(unused_mut)]
#![allow(unused_variables)]

use crate::dtype::{IntegerDataType, NumericDataType};
use std::hint::assert_unchecked;

use paste::paste;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub(super) fn get_sum_of_products_function<const N: usize, T: SumOfProductsType>(strides: &[usize; N])
                                                                                 -> unsafe fn(ptrs: &[*mut T; N], stride: &[usize; N], count: usize) {
    if N == 2 { // 1 operand + 1 output
        if strides[0] == 1 && strides[1] == 0 {
            // return <T as EinsumDataType>::operand_strides_1_out_stride_0;
        }
    }

    if N == 3 { // 2 operands + 1 output
        let mut code = if strides[0] == 0 { 0 } else { if strides[0] == 1 { 4 } else { 8 } };
        code += if strides[1] == 0 { 0 } else { if strides[1] == 1 { 2 } else { 8 } };
        code += if strides[2] == 0 { 0 } else { if strides[2] == 1 { 1 } else { 8 } };

        match code {
            2 => { return <T as SumOfProductsType>::operand_strides_0_1_out_stride_0; }
            3 => { return <T as SumOfProductsType>::operand_strides_0_1_out_stride_1; }
            4 => { return <T as SumOfProductsType>::operand_strides_1_0_out_stride_0; }
            5 => { return <T as SumOfProductsType>::operand_strides_1_0_out_stride_1; }
            6 => { return <T as SumOfProductsType>::operand_strides_1_1_out_stride_0; }
            7 => { return <T as SumOfProductsType>::operand_strides_1_1_out_stride_1; }
            _ => {}
        }
    }

    if strides[N - 1] == 0 {
        return <T as SumOfProductsType>::out_stride_0;
    }

    <T as SumOfProductsType>::generic
}

// called when the number of operands cannot be provided as a const generic
pub(super) fn get_sum_of_products_function_generic_nops<T: SumOfProductsType>(strides: &[usize])
                                                                              -> unsafe fn(ptrs: &[*mut T], stride: &[usize], count: usize) {
    let nops = strides.len() - 1;

    if strides[nops - 1] == 0 {
        return match nops {
            3 => { <T as SumOfProductsType>::operand_strides_n_n_n_out_stride_0 },
            _ => { <T as SumOfProductsType>::operand_strides_nx_out_stride_0 }
        }
    }

    <T as SumOfProductsType>::generic_unknown_nops
}

pub(super) trait SumOfProductsType: NumericDataType {
    #[inline(always)]
    unsafe fn generic_unknown_nops(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        let nops = ptrs.len();
        assert_unchecked(count > 0);
        assert_unchecked(nops > 0);

        let dst = ptrs[nops - 1];
        let ptrs = &ptrs[0..nops - 1];

        let mut k = count;
        while k != 0 {
            k -= 1;

            let dst = dst.add(k * strides[nops - 1]);
            *dst += ptrs.iter().zip(strides.iter())
                        .map(|(ptr, stride)| *ptr.add(k * stride))
                        .product();
        }
    }

    #[inline(always)]
    unsafe fn generic<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(count > 0);

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
    unsafe fn operand_strides_n_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        const NOPS: usize = 3;
        assert_unchecked(count > 0);
        assert_unchecked(ptrs.len() - 1 == NOPS);

        let dst = ptrs[NOPS];
        let ptrs = &ptrs[..NOPS];

        let mut sum = Self::default();

        let mut k = count;
        while k != 0 {
            k -= 1;

            let mut product = Self::one();
            for i in 0..NOPS {
                product *= *ptrs[i].add(k * strides[i]);
            }
            sum += product;
        }

        *dst += sum;
    }

    #[inline(always)]
    unsafe fn operand_strides_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        const NOPS: usize = 2;
        assert_unchecked(count > 0);
        assert_unchecked(ptrs.len() - 1 == NOPS);

        let dst = ptrs[NOPS];
        let mut ptrs = [ptrs[0], ptrs[1]];

        let mut sum = Self::default();

        let mut k = count;
        while k != 0 {
            sum += (*ptrs[0]) * (*ptrs[1]);

            k -= 1;
            ptrs[0] = ptrs[0].add(strides[0]);
            ptrs[1] = ptrs[1].add(strides[1]);
        }

        *dst += sum;
    }

    #[inline(always)]
    unsafe fn operand_strides_nx_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        let nops = ptrs.len() - 1;
        assert_unchecked(count > 0);

        let dst = ptrs[nops];
        let ptrs = &ptrs[..nops];

        let mut sum = Self::default();

        let mut k = count;
        while k != 0 {
            k -= 1;

            let mut product = Self::one();
            for i in 0..nops {
                product *= *ptrs[i].add(k * strides[i]);
            }
            sum += product;
        }

        *dst += sum;
    }

    #[inline(always)]
    unsafe fn out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(count > 0);

        let dst = ptrs[N - 1];
        let ptrs = &ptrs[0..N - 1];

        let mut k = count;
        while k != 0 {
            k -= 1;

            *dst += ptrs.iter().zip(strides.iter())
                        .map(|(ptr, stride)| *ptr.add(k * stride))
                        .product();
        }
    }

    #[inline(always)]
    unsafe fn sum_of_products_muladd<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(count > 0);

        let mut dst = ptrs[N - 1];

        let scalar = *ptrs[0];
        let mut data = ptrs[1];

        for _ in 0..count {
            *dst = scalar.mul_add(*data, *dst);
            dst = dst.add(1);
            data = data.add(1);
        }
    }

    #[inline(always)]
    unsafe fn sum_of_scaled_array<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(count > 0);

        let mut dst = ptrs[N - 1];
        let scalar = *ptrs[0];
        let mut data = ptrs[1];

        let mut sum = Self::zero();
        for i in 0..count {
            sum += *data.add(i);
        }

        *dst = scalar.mul_add(sum, *dst);
    }

    #[inline(always)]
    unsafe fn operand_strides_0_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);
        Self::sum_of_scaled_array(&ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn operand_strides_0_1_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);
        Self::sum_of_products_muladd(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn operand_strides_1_0_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);

        let mut ptrs = *ptrs;
        let tmp = ptrs[0];
        ptrs[0] = ptrs[1];
        ptrs[1] = tmp;

        Self::sum_of_scaled_array(&ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn operand_strides_1_0_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);

        let mut ptrs = *ptrs;
        let tmp = ptrs[0];
        ptrs[0] = ptrs[1];
        ptrs[1] = tmp;

        Self::sum_of_products_muladd(&ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn operand_strides_1_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);
        assert_unchecked(count > 0);

        let mut dst = ptrs[N - 1];

        let mut data0 = ptrs[0];
        let mut data1 = ptrs[1];
        let mut sum = Self::default();

        for i in 0..count {
            sum += (*data0) * (*data1);
            data0 = data0.add(1);
            data1 = data1.add(1);
        }

        *dst += sum;
    }

    #[inline(always)]
    unsafe fn operand_strides_1_1_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);
        assert_unchecked(count > 0);

        let mut data0 = ptrs[0];
        let mut data1 = ptrs[1];
        let mut dst = ptrs[2];

        for i in 0..count {
            *dst += (*data0) * (*data1);

            dst = dst.add(1);
            data0 = data0.add(1);
            data1 = data1.add(1);
        }

        // TODO specialised SIMD implementation
    }
}

impl<T: IntegerDataType> SumOfProductsType for T {}

macro_rules! simd_kernel_for_dtype {
    ($dtype:ty, $nlanes:literal,
    $ptrs:ident, $strides:ident, $count:ident, $dst:ident, $lanes:ident,

    $simd_load:ident, $vload:expr,
    $simd_store:ident, $vstore:expr,
    $simd_add:ident, $vadd:expr,
    $simd_mul:ident, $vmul:expr,
    $simd_muladd:ident, $vmuladd:expr,
    $simd_sum:ident, $vaddv:expr,
    $simd_dup:ident, $vdup:expr,

    $($func_name:ident, { $($body:tt)* };)+) => { paste! {
            #[cfg(not(target_arch = "aarch64"))]
            impl SumOfProductsType for $dtype {}

            #[cfg(target_arch = "aarch64")]
            impl SumOfProductsType for $dtype {
                $(
                #[inline(always)]
                unsafe fn $func_name<const N: usize>($ptrs: &[*mut Self; N], $strides: &[usize; N], mut $count: usize) {
                    assert_unchecked($count > 0);

                    const $lanes: usize = $nlanes;
                    let $simd_load = $vload;
                    let $simd_store = $vstore;
                    let $simd_add = $vadd;
                    let $simd_mul = $vmul;
                    let $simd_muladd = $vmuladd;
                    let $simd_sum = $vaddv;
                    let $simd_dup = $vdup;

                    let mut $dst = $ptrs[N - 1];

                    $($body)*
                }
                )+
            }
        }
    }
}

macro_rules! simd_kernel {
    ($ptrs:ident, $strides:ident, $count:ident, $dst:ident,
    $lanes:ident, $simd_load:ident, $simd_store:ident, $simd_add:ident, $simd_mul:ident, $simd_muladd:ident, $simd_sum:ident, $simd_dup:ident,
    $($func_name:ident, { $($body:tt)* },)+) => {

        simd_kernel_for_dtype!(f32, 4, $ptrs, $strides, $count, $dst, $lanes,
                                $simd_load, vld1q_f32, $simd_store, vst1q_f32, $simd_add, vaddq_f32, $simd_mul, vmulq_f32,
                                $simd_muladd, vfmaq_f32, $simd_sum, vaddvq_f32, $simd_dup, vdupq_n_f32,
                                $($func_name, { $($body)* };)+);

        simd_kernel_for_dtype!(f64, 2, $ptrs, $strides, $count, $dst, $lanes,
                                $simd_load, vld1q_f64, $simd_store, vst1q_f64, $simd_add, vaddq_f64, $simd_mul, vmulq_f64,
                                $simd_muladd, vfmaq_f64, $simd_sum, vaddvq_f64, $simd_dup, vdupq_n_f64,
                                $($func_name, { $($body)* };)+);
    };
}

simd_kernel!(ptrs, strides, count, dst, LANES, simd_load, simd_store, simd_add, simd_mul, simd_muladd, simd_sum, simd_dup,
    sum_of_products_muladd, {
        let value0 = *ptrs[0];
        let value0x = simd_dup(value0);
        let mut data1 = ptrs[1];

        while count >= 4 * LANES {
            let a = simd_load(data1.add(0 * LANES));
            let b = simd_load(data1.add(1 * LANES));
            let c = simd_load(data1.add(2 * LANES));
            let d = simd_load(data1.add(3 * LANES));

            let a_dst = simd_load(dst.add(0 * LANES));
            let b_dst = simd_load(dst.add(1 * LANES));
            let c_dst = simd_load(dst.add(2 * LANES));
            let d_dst = simd_load(dst.add(3 * LANES));

            let a_out = simd_muladd(a_dst, value0x, a);
            let b_out = simd_muladd(b_dst, value0x, b);
            let c_out = simd_muladd(c_dst, value0x, c);
            let d_out = simd_muladd(d_dst, value0x, d);

            simd_store(dst.add(0 * LANES), a_out);
            simd_store(dst.add(1 * LANES), b_out);
            simd_store(dst.add(2 * LANES), c_out);
            simd_store(dst.add(3 * LANES), d_out);

            count -= 4 * LANES;
            dst = dst.add(4 * LANES);
            data1 = data1.add(4 * LANES);
        }

        while count >= LANES {
            let a = simd_load(data1);
            let a_dst = simd_load(dst);
            let a_out = simd_muladd(a_dst, value0x, a);

            simd_store(dst, a_out);

            count -= LANES;
            dst = dst.add(LANES);
            data1 = data1.add(LANES);
        }

        for _ in 0..count {
            *dst = value0.mul_add(*data1, *dst);
            dst = dst.add(1);
            data1 = data1.add(1);
        }
    },

    sum_of_scaled_array, {
        let value0 = *ptrs[0];
        let mut data1 = ptrs[1];

        let mut sum = simd_dup(Self::default());

        while count >= 4 * LANES {
            let a = simd_load(data1.add(0 * LANES));
            let b = simd_load(data1.add(1 * LANES));
            let c = simd_load(data1.add(2 * LANES));
            let d = simd_load(data1.add(3 * LANES));

            let ab = simd_add(a, b);
            let cd = simd_add(c, d);
            sum = simd_add(sum, simd_add(ab, cd));

            count -= 4 * LANES;
            data1 = data1.add(4 * LANES);
        }

        while count >= LANES {
            let a = simd_load(data1);
            sum = simd_add(sum, a);

            count -= LANES;
            data1 = data1.add(LANES);
        }

        let mut sum = simd_sum(sum);
        for i in 0..count {
            sum += *data1.add(i);
        }

        *dst = value0.mul_add(sum, *dst);
    },

    operand_strides_1_1_out_stride_0, {
        let mut data0 = ptrs[0];
        let mut data1 = ptrs[1];
        let mut sum = simd_dup(Self::default());

        while count >= 4 * LANES {
            let a0 = simd_load(data0.add(0 * LANES));
            let b0 = simd_load(data1.add(0 * LANES));

            let a1 = simd_load(data0.add(1 * LANES));
            let b1 = simd_load(data1.add(1 * LANES));

            let a2 = simd_load(data0.add(2 * LANES));
            let b2 = simd_load(data1.add(2 * LANES));

            let a3 = simd_load(data0.add(3 * LANES));
            let b3 = simd_load(data1.add(3 * LANES));

            let ab0 = simd_mul(a0, b0);
            let ab1 = simd_mul(a1, b1);
            let ab2 = simd_mul(a2, b2);
            let ab3 = simd_mul(a3, b3);

            let ab01 = simd_add(ab0, ab1);
            let ab23 = simd_add(ab2, ab3);
            let ab0123 = simd_add(ab01, ab23);
            
            sum = simd_add(sum, ab0123);

            count -= 4 * LANES;
            data0 = data0.add(4 * LANES);
            data1 = data1.add(4 * LANES);
        }

        while count >= LANES {
            let a = simd_load(data0);
            let b = simd_load(data1);
            sum = simd_muladd(sum, a, b);
        
            count -= LANES;
            data0 = data0.add(LANES);
            data1 = data1.add(LANES);
        }

        let mut sum = simd_sum(sum);
        for i in 0..count {
            sum = (*data0).mul_add(*data1, sum);
            data0 = data0.add(1);
            data1 = data1.add(1);
        }

        *dst += sum;
    },
);
