#![allow(unused_mut)]
#![allow(unused_variables)]

use crate::accelerate::simd::SIMD;
use crate::dtype::{IntegerDataType, NumericDataType};
use std::hint::assert_unchecked;

use paste::paste;

pub(super) fn get_sum_of_products_function<const N: usize, T: SumOfProductsType>(strides: &[usize; N])
                                                                                 -> unsafe fn(ptrs: &[*mut T; N], stride: &[usize; N], count: usize) {
    if N == 2 { // 1 operand + 1 output
        if strides[0] == 1 && strides[1] == 0 {
            // return <T as EinsumDataType>::operand_strides_1_out_stride_0; // can be implemented later if needed
        }
    }

    if N == 3 { // 2 operands + 1 output
        let mut code = if strides[0] == 0 { 0 } else if strides[0] == 1 { 4 } else { 8 };
        code += if strides[1] == 0 { 0 } else if strides[1] == 1 { 2 } else { 8 };
        code += if strides[2] == 0 { 0 } else if strides[2] == 1 { 1 } else { 8 };

        match code {
            2 => { return <T as SumOfProductsType>::sum_of_products_in_strides_0_1_out_stride_0; }
            3 => { return <T as SumOfProductsType>::sum_of_products_in_strides_0_1_out_stride_1; }
            4 => { return <T as SumOfProductsType>::sum_of_products_in_strides_1_0_out_stride_0; }
            5 => { return <T as SumOfProductsType>::sum_of_products_in_strides_1_0_out_stride_1; }
            6 => { return <T as SumOfProductsType>::sum_of_products_in_strides_1_1_out_stride_0; }
            7 => { return <T as SumOfProductsType>::sum_of_products_in_strides_1_1_out_stride_1; }
            _ => {}
        }
    }

    if strides[N - 1] == 0 {
        return <T as SumOfProductsType>::sum_of_products_out_stride_0;
    }

    <T as SumOfProductsType>::sum_of_products_generic
}

// called when the number of operands cannot be provided as a const generic
pub(super) fn get_sum_of_products_function_generic_nops<T: SumOfProductsType>(strides: &[usize])
                                                                              -> unsafe fn(ptrs: &[*mut T], stride: &[usize], count: usize) {
    let nops = strides.len() - 1;

    if strides[nops] == 0 {
        return match nops {
            3 => { <T as SumOfProductsType>::sum_of_products_in_strides_n_n_n_out_stride_0 },
            _ => { <T as SumOfProductsType>::sum_of_products_out_stride_0_ }
        }
    }

    <T as SumOfProductsType>::sum_of_products_generic_
}


pub(crate) trait SumOfProductsType: NumericDataType {
    #[inline(always)]
    unsafe fn sum_of_products_generic_(ptrs: &[*mut Self], strides: &[usize], count: usize) {
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
    unsafe fn sum_of_products_generic<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        Self::sum_of_products_generic_(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_in_strides_n_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        assert_unchecked(ptrs.len() == 4);
        assert_unchecked(strides.len() == 4);
        Self::sum_of_products_out_stride_0_(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_in_strides_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        assert_unchecked(ptrs.len() == 3);
        assert_unchecked(strides.len() == 3);
        Self::sum_of_products_out_stride_0_(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        Self::sum_of_products_out_stride_0_(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_out_stride_0_(ptrs: &[*mut Self], strides: &[usize], count: usize) {
        assert_unchecked(count > 0);
        assert_unchecked(strides[strides.len() - 1] == 0);
        
        let nops = ptrs.len() - 1;
        let dst = ptrs[nops];
        let ptrs = &ptrs[..nops];

        let mut sum = Self::default();

        let mut k = count;
        while k != 0 {
            k -= 1;

            sum += ptrs.iter().zip(strides.iter())
                       .map(|(ptr, stride)| *ptr.add(k * stride))
                       .product();
        }

        *dst += sum;
    }

    #[inline(always)]
    unsafe fn sum_of_products_muladd<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(count > 0);
        assert_unchecked(N == 3);

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
    unsafe fn sum_of_products_in_strides_0_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);
        assert_unchecked(strides[0] == 0);
        assert_unchecked(strides[1] == 1);
        assert_unchecked(strides[2] == 0);
        Self::sum_of_scaled_array(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_in_strides_0_1_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(N == 3);
        assert_unchecked(strides[0] == 0);
        assert_unchecked(strides[1] == 1);
        assert_unchecked(strides[2] == 1);
        Self::sum_of_products_muladd(ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_in_strides_1_0_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(strides[0] == 1);
        assert_unchecked(strides[1] == 0);
        assert_unchecked(strides[2] == 0);
        assert_unchecked(N == 3);

        let mut ptrs = *ptrs;
        ptrs.swap(0, 1);

        Self::sum_of_scaled_array(&ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_in_strides_1_0_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(strides[0] == 1);
        assert_unchecked(strides[1] == 0);
        assert_unchecked(strides[2] == 1);
        assert_unchecked(N == 3);

        let mut ptrs = *ptrs;
        ptrs.swap(0, 1);

        Self::sum_of_products_muladd(&ptrs, strides, count);
    }

    #[inline(always)]
    unsafe fn sum_of_products_in_strides_1_1_out_stride_0<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(strides[0] == 1);
        assert_unchecked(strides[1] == 1);
        assert_unchecked(strides[2] == 0);
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
    unsafe fn sum_of_products_in_strides_1_1_out_stride_1<const N: usize>(ptrs: &[*mut Self; N], strides: &[usize; N], count: usize) {
        assert_unchecked(strides[0] == 1);
        assert_unchecked(strides[1] == 1);
        assert_unchecked(strides[2] == 1);
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

macro_rules! simd_sum_of_products_kernels {
    ($ptrs:ident, $strides:ident, $count:ident, $dst:ident, $($func_name:ident, { $($body:tt)* };)+) => {
        paste! {$(
            #[cfg(neon_simd)]
            #[inline(always)]
            #[allow(clippy::erasing_op)]
            #[allow(clippy::identity_op)]
            unsafe fn $func_name<const N: usize>($ptrs: &[*mut Self; N], $strides: &[usize; N], mut $count: usize) {
                assert_unchecked($count > 0);
                let mut $dst = $ptrs[N - 1];

                $($body)*
            }
        )+}
    }
}

macro_rules! accelerated_sum_of_products {
    ($ptrs:ident, $strides:ident, $count:ident, $dst:ident, $($func_name:ident, { $($body:tt)* },)+) => {

        impl SumOfProductsType for f32 {
            simd_sum_of_products_kernels!($ptrs, $strides, $count, $dst, $($func_name, { $($body)* };)+);

            #[cfg(apple_vdsp)]
            #[inline(always)]
            unsafe fn sum_of_products_in_strides_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
                use crate::accelerate::vdsp::vDSP_dotpr;
                vDSP_dotpr(ptrs[0], strides[0] as isize, ptrs[1], strides[1] as isize, ptrs[2], count as isize);
            }

            #[cfg(all(not(apple_vdsp), not(neon_simd), blas))]
            #[inline(always)]
            unsafe fn sum_of_products_in_strides_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
                use crate::accelerate::cblas::cblas_sdot;
                *ptrs[2] = cblas_sdot(count as i32, ptrs[0], strides[0] as i32, ptrs[1], strides[1] as i32);
            }
        }

        impl SumOfProductsType for f64 {
            simd_sum_of_products_kernels!($ptrs, $strides, $count, $dst, $($func_name, { $($body)* };)+);

            #[cfg(apple_vdsp)]
            #[inline(always)]
            unsafe fn sum_of_products_in_strides_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
                use crate::accelerate::vdsp::vDSP_dotprD;
                vDSP_dotprD(ptrs[0], strides[0] as isize, ptrs[1], strides[1] as isize, ptrs[2], count as isize);
            }
            
            #[cfg(all(not(apple_vdsp), not(neon_simd), blas))]
            #[inline(always)]
            unsafe fn sum_of_products_in_strides_n_n_out_stride_0(ptrs: &[*mut Self], strides: &[usize], count: usize) {
                use crate::accelerate::cblas::cblas_ddot;
                *ptrs[2] = cblas_ddot(count as i32, ptrs[0], strides[0] as i32, ptrs[1], strides[1] as i32);
            }
        }
    };
}

accelerated_sum_of_products!(ptrs, strides, count, dst,
    sum_of_products_muladd, {
        let value0 = *ptrs[0];
        let value0x = Self::simd_dup(value0);
        let mut data1 = ptrs[1];

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(data1.add(0 * Self::LANES));
            let b = Self::simd_load(data1.add(1 * Self::LANES));
            let c = Self::simd_load(data1.add(2 * Self::LANES));
            let d = Self::simd_load(data1.add(3 * Self::LANES));

            let a_dst = Self::simd_load(dst.add(0 * Self::LANES));
            let b_dst = Self::simd_load(dst.add(1 * Self::LANES));
            let c_dst = Self::simd_load(dst.add(2 * Self::LANES));
            let d_dst = Self::simd_load(dst.add(3 * Self::LANES));

            let a_out = Self::simd_muladd(a_dst, value0x, a);
            let b_out = Self::simd_muladd(b_dst, value0x, b);
            let c_out = Self::simd_muladd(c_dst, value0x, c);
            let d_out = Self::simd_muladd(d_dst, value0x, d);

            Self::simd_store(dst.add(0 * Self::LANES), a_out);
            Self::simd_store(dst.add(1 * Self::LANES), b_out);
            Self::simd_store(dst.add(2 * Self::LANES), c_out);
            Self::simd_store(dst.add(3 * Self::LANES), d_out);

            count -= 4 * Self::LANES;
            dst = dst.add(4 * Self::LANES);
            data1 = data1.add(4 * Self::LANES);
        }

        while count >= Self::LANES {
            let a = Self::simd_load(data1);
            let a_dst = Self::simd_load(dst);
            let a_out = Self::simd_muladd(a_dst, value0x, a);

            Self::simd_store(dst, a_out);

            count -= Self::LANES;
            dst = dst.add(Self::LANES);
            data1 = data1.add(Self::LANES);
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

        let mut sum = Self::simd_dup(Self::default());

        while count >= 4 * Self::LANES {
            let a = Self::simd_load(data1.add(0 * Self::LANES));
            let b = Self::simd_load(data1.add(1 * Self::LANES));
            let c = Self::simd_load(data1.add(2 * Self::LANES));
            let d = Self::simd_load(data1.add(3 * Self::LANES));

            let ab = Self::simd_add(a, b);
            let cd = Self::simd_add(c, d);
            sum = Self::simd_add(sum, Self::simd_add(ab, cd));

            count -= 4 * Self::LANES;
            data1 = data1.add(4 * Self::LANES);
        }

        while count >= Self::LANES {
            let a = Self::simd_load(data1);
            sum = Self::simd_add(sum, a);

            count -= Self::LANES;
            data1 = data1.add(Self::LANES);
        }

        let mut sum = Self::simd_sum(sum);
        for i in 0..count {
            sum += *data1.add(i);
        }

        *dst = value0.mul_add(sum, *dst);
    },

    sum_of_products_in_strides_1_1_out_stride_0, {
        let mut data0 = ptrs[0];
        let mut data1 = ptrs[1];
        let mut sum = Self::simd_dup(Self::default());

        while count >= 4 * Self::LANES {
            let a0 = Self::simd_load(data0.add(0 * Self::LANES));
            let b0 = Self::simd_load(data1.add(0 * Self::LANES));

            let a1 = Self::simd_load(data0.add(1 * Self::LANES));
            let b1 = Self::simd_load(data1.add(1 * Self::LANES));

            let a2 = Self::simd_load(data0.add(2 * Self::LANES));
            let b2 = Self::simd_load(data1.add(2 * Self::LANES));

            let a3 = Self::simd_load(data0.add(3 * Self::LANES));
            let b3 = Self::simd_load(data1.add(3 * Self::LANES));

            let ab0 = Self::simd_mul(a0, b0);
            let ab1 = Self::simd_mul(a1, b1);
            let ab2 = Self::simd_mul(a2, b2);
            let ab3 = Self::simd_mul(a3, b3);

            let ab01 = Self::simd_add(ab0, ab1);
            let ab23 = Self::simd_add(ab2, ab3);
            let ab0123 = Self::simd_add(ab01, ab23);
            
            sum = Self::simd_add(sum, ab0123);

            count -= 4 * Self::LANES;
            data0 = data0.add(4 * Self::LANES);
            data1 = data1.add(4 * Self::LANES);
        }

        while count >= Self::LANES {
            let a = Self::simd_load(data0);
            let b = Self::simd_load(data1);
            sum = Self::simd_muladd(sum, a, b);
        
            count -= Self::LANES;
            data0 = data0.add(Self::LANES);
            data1 = data1.add(Self::LANES);
        }

        let mut sum = Self::simd_sum(sum);
        for i in 0..count {
            sum = (*data0).mul_add(*data1, sum);
            data0 = data0.add(1);
            data1 = data1.add(1);
        }

        *dst += sum;
    },
);
