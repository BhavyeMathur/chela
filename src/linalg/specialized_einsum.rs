use std::hint::assert_unchecked;
use crate::linalg::sum_of_products::*;
use crate::dtype::NumericDataType;
use crate::linalg::sum_of_products::{get_sum_of_products_function, SumOfProducts};
use crate::{dispatch_einsum_func, Tensor};

unsafe fn einsum_2operands_3labels_3outputdims<T: NumericDataType, F: SumOfProducts<2, T>>(op1: *const T,
                                                                                           op2: *const T,
                                                                                           mut dst: *mut T,
                                                                                           strides1: &[usize; 3],
                                                                                           strides2: &[usize; 3],
                                                                                           iter_shape: &[usize; 3]) {
    for i in 0..iter_shape[0] {
        for j in 0..iter_shape[1] {
            for k in 0..iter_shape[2] {
                *dst = (*op1.add(i * strides1[0] + j * strides1[1] + k * strides1[2]))
                    * (*op2.add(i * strides2[0] + j * strides2[1] + k * strides2[2]));
                dst = dst.add(1);
            }
        }
    }
}

unsafe fn einsum_2operands_3labels_2outputdims<T: NumericDataType, F: SumOfProducts<2, T>>(op1: *const T,
                                                                                           op2: *const T,
                                                                                           mut dst: *mut T,
                                                                                           strides1: &[usize; 3],
                                                                                           strides2: &[usize; 3],
                                                                                           iter_shape: &[usize; 3]) {
    for i in 0..iter_shape[0] {
        for j in 0..iter_shape[1] {
            let mut sum = T::zero();

            for k in 0..iter_shape[2] {
                sum += (*op1.add(i * strides1[0] + j * strides1[1] + k * strides1[2]))
                    * (*op2.add(i * strides2[0] + j * strides2[1] + k * strides2[2]))
            }

            *dst = sum;
            dst = dst.add(1);
        }
    }
}

unsafe fn einsum_2operands_3labels_1outputdims<T: NumericDataType, F: SumOfProducts<2, T>>(op1: *const T,
                                                                                           op2: *const T,
                                                                                           mut dst: *mut T,
                                                                                           strides1: &[usize; 3],
                                                                                           strides2: &[usize; 3],
                                                                                           iter_shape: &[usize; 3]) {
    for i in 0..iter_shape[0] {
        let mut sum = T::zero();

        for j in 0..iter_shape[1] {
            for k in 0..iter_shape[2] {
                sum += (*op1.add(i * strides1[0] + j * strides1[1] + k * strides1[2]))
                    * (*op2.add(i * strides2[0] + j * strides2[1] + k * strides2[2]))
            }
        }

        *dst = sum;
        dst = dst.add(1);
    }
}

#[inline(never)]
unsafe fn einsum_2operands_3labels_0outputdims<T: NumericDataType, F: SumOfProducts<2, T>>(op1: *const T,
                                                                                           op2: *const T,
                                                                                           dst: *mut T,
                                                                                           strides1: &[usize; 3],
                                                                                           strides2: &[usize; 3],
                                                                                           iter_shape: &[usize; 3]) {
    let mut sum = T::zero();

    let inner_loop_strides = [strides1[2], strides2[2]];

    for i in 0..iter_shape[0] {
        for j in 0..iter_shape[1] {
            let ptr1 = op1.add(i * strides1[0] + j * strides1[1]);
            let ptr2 = op2.add(i * strides2[0] + j * strides2[1]);

            sum += F::call(&[ptr1, ptr2], &inner_loop_strides, iter_shape[2]);
        }
    }

    *dst = sum;
}

pub(super) fn einsum_2operands_3labels<'b, T: NumericDataType>(operand1: &Tensor<T>,
                                                               operand2: &Tensor<T>,
                                                               strides1: &[usize; 3],
                                                               strides2: &[usize; 3],
                                                               iter_shape: &[usize; 3],
                                                               mut output: Vec<T>,
                                                               output_shape: Vec<usize>) -> Tensor<'b, T> {
    unsafe {
        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);
    }

    let op1 = operand1.ptr.as_ptr() as *const T;
    let op2 = operand2.ptr.as_ptr() as *const T;
    let dst = output.as_mut_ptr();

    let output_dims = output_shape.len();
    let sum_of_products = get_sum_of_products_function([strides1[2], strides2[2]]);

    unsafe {
        if output_dims == 0 {
            dispatch_einsum_func!( sum_of_products, einsum_2operands_3labels_0outputdims,
                op1, op2, dst, strides1, strides2, iter_shape
            );
        } else if output_dims == 1 {
            dispatch_einsum_func!( sum_of_products, einsum_2operands_3labels_1outputdims,
                op1, op2, dst, strides1, strides2, iter_shape
            );
        } else if output_dims == 2 {
            dispatch_einsum_func!( sum_of_products, einsum_2operands_3labels_2outputdims,
                op1, op2, dst, strides1, strides2, iter_shape
            );
        } else if output_dims == 3 {
            dispatch_einsum_func!( sum_of_products, einsum_2operands_3labels_3outputdims,
                op1, op2, dst, strides1, strides2, iter_shape
            );
        }
    }

    unsafe { Tensor::from_contiguous_owned_buffer(output_shape, output) }
}
