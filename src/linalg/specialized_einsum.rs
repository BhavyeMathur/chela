use crate::linalg::sum_of_products::*;
use crate::Tensor;
use std::hint::assert_unchecked;


pub(super) fn einsum_2operands_3labels<'b, T: EinsumDataType>(operand1: &Tensor<T>,
                                                              operand2: &Tensor<T>,
                                                              strides1: &[usize; 3],
                                                              strides2: &[usize; 3],
                                                              output_strides: &[usize; 3],
                                                              iter_shape: &[usize; 3],
                                                              mut output: Vec<T>,
                                                              output_shape: Vec<usize>) -> Tensor<'b, T>
{
    unsafe {
        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);
    }

    let op1 = operand1.ptr.as_ptr() as *const T;
    let op2 = operand2.ptr.as_ptr() as *const T;
    let dst = output.as_mut_ptr();

    let inner_loop_strides = [strides1[2], strides2[2]];
    let sum_of_products = get_sum_of_products_function(inner_loop_strides, output_strides[2]);

    unsafe {
        for i in 0..iter_shape[0] {
            for j in 0..iter_shape[1] {
                let ptr1 = op1.add(i * strides1[0] + j * strides1[1]);
                let ptr2 = op2.add(i * strides2[0] + j * strides2[1]);
                let dst = dst.add(i * output_strides[0] + j * output_strides[1]);

                sum_of_products(&[ptr1, ptr2], &inner_loop_strides, output_strides[2], iter_shape[2], dst);
            }
        }
    }

    unsafe { Tensor::from_contiguous_owned_buffer(output_shape, output) }
}
