use crate::linalg::sum_of_products::*;
use crate::Tensor;
use std::hint::assert_unchecked;


pub(super) fn einsum_2operands_3labels<'b, T: EinsumDataType>(operand1: &Tensor<T>,
                                                              operand2: &Tensor<T>,
                                                              strides_dim0: &[usize; 3],
                                                              strides_dim1: &[usize; 3],
                                                              strides_dim2: &[usize; 3],
                                                              iter_shape: &[usize; 3],
                                                              mut output: Vec<T>,
                                                              output_shape: Vec<usize>) -> Tensor<'b, T>
{
    unsafe {
        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);
    }

    let op1 = operand1.ptr.as_ptr();
    let op2 = operand2.ptr.as_ptr();
    let dst = output.as_mut_ptr();
    
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        for i in 0..iter_shape[2] {
            for j in 0..iter_shape[1] {
                let ptr1 = op1.add(i * strides_dim2[0] + j * strides_dim1[0]);
                let ptr2 = op2.add(i * strides_dim2[1] + j * strides_dim1[1]);
                let dst = dst.add(i * strides_dim2[2] + j * strides_dim1[2]);

                sum_of_products(&[ptr1, ptr2, dst], &strides_dim0, iter_shape[0]);
            }
        }
    }

    unsafe { Tensor::from_contiguous_owned_buffer(output_shape, output) }
}
