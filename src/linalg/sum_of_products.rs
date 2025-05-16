use std::hint::assert_unchecked;
use crate::dtype::NumericDataType;
use crate::tensor::MAX_DIMS;

pub(super) fn get_sum_of_products_function<const N: usize>(strides: [usize; N]) -> SumOfProductFunc {
    if N == 2 {
        let mut code = if strides[0] == 0 { 0 } else { if strides[0] == 1 { 4 } else { 8 } };
        code += if strides[1] == 0 { 0 } else { if strides[1] == 1 { 2 } else { 8 } };
        // code += if strides[2] == 0 { 0 } else { if strides[2] == 1 { 1 } else { 8 } };  // NumPy stores the output's stride as element 2

        if code == 2 {
            return SumOfProductFunc::Stride0ContiguousOutstride0Two;
        }
    }

    SumOfProductFunc::Generic
}

pub(super) enum SumOfProductFunc {
    Generic,
    Stride0ContiguousOutstride0Two
}

pub(super) struct SumOfProductsGeneric;
pub(super) struct Stride0ContiguousOutstride0Two;

#[macro_export]
macro_rules! dispatch_einsum_func {
    (
        $dispatch_enum:expr,
        $einsum_fn:ident,
        $( $arg:expr ),* $(,)?
    ) => {
        match $dispatch_enum {
            SumOfProductFunc::Generic => {
                $einsum_fn::<T, SumOfProductsGeneric>($( $arg ),*);
            }
            SumOfProductFunc::Stride0ContiguousOutstride0Two => {
                $einsum_fn::<T, Stride0ContiguousOutstride0Two>($( $arg ),*);
            }
        }
    };
}

pub(super) trait SumOfProducts<const N: usize, T: NumericDataType> {
    unsafe fn call(ptrs: &[*const T; N], strides: &[usize; N], count: usize) -> T;
}

impl<const N: usize, T: NumericDataType> SumOfProducts<N, T> for SumOfProductsGeneric {
    #[inline(always)]
    unsafe fn call(ptrs: &[*const T; N], strides: &[usize; N], count: usize) -> T {
        assert_unchecked(count > 0);
        assert_unchecked(N > 0 && N <= MAX_DIMS);

        let mut sum = T::zero();

        let mut k = count;
        while k != 0 {
            k -= 1;
            sum += ptrs.iter().zip(strides.iter())
                       .map(|(ptr, stride)| *ptr.add(k * stride))
                       .product();
        }

        sum
    }
}

impl<const N: usize, T: NumericDataType> SumOfProducts<N, T> for Stride0ContiguousOutstride0Two {
    #[inline(always)]
    unsafe fn call(ptrs: &[*const T; N], _: &[usize; N], count: usize) -> T {
        assert_unchecked(count > 0);
        assert_unchecked(N == 2);

        let value0 = *ptrs[0];
        let data1 = std::slice::from_raw_parts(ptrs[1], count);
        value0 * data1.iter().copied().sum()
    }
}
