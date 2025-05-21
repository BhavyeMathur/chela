use crate::linalg::sum_of_products::*;
use crate::Tensor;
use std::hint::assert_unchecked;


pub(super) fn einsum_1operand_2labels<'b, T: SumOfProductsType>(operand: &Tensor<T>,
                                                                strides_dim0: &[usize; 2],
                                                                strides_dim1: &[usize; 2],
                                                                iter_shape: &[usize; 2],
                                                                mut output: Vec<T>,
                                                                output_shape: Vec<usize>) -> Tensor<'b, T>
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op = operand.mut_ptr();
        let dst = output.as_mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        
        for i in 0..iter_shape[1] {
            let src = op.add(i * strides_dim1[0]);
            let dst = dst.add(i * strides_dim1[1]);

            sum_of_products(&[src, dst], &strides_dim0, iter_shape[0]);
        }

        Tensor::from_contiguous_owned_buffer(output_shape, output)
    }
}

pub(super) fn einsum_1operand_3labels<'b, T: SumOfProductsType>(operand: &Tensor<T>,
                                                                strides_dim0: &[usize; 2],
                                                                strides_dim1: &[usize; 2],
                                                                strides_dim2: &[usize; 2],
                                                                iter_shape: &[usize; 3],
                                                                mut output: Vec<T>,
                                                                output_shape: Vec<usize>) -> Tensor<'b, T>
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op = operand.mut_ptr();
        let dst = output.as_mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);
        
        for i in 0..iter_shape[2] {
            for j in 0..iter_shape[1] {
                let src = op.add(i * strides_dim2[0] + j * strides_dim1[0]);
                let dst = dst.add(i * strides_dim2[1] + j * strides_dim1[1]);

                sum_of_products(&[src, dst], &strides_dim0, iter_shape[0]);
            }
        }

        Tensor::from_contiguous_owned_buffer(output_shape, output)
    }
}

pub(super) fn einsum_2operands_2labels<'b, T: SumOfProductsType>(operand1: &Tensor<T>,
                                                                 operand2: &Tensor<T>,
                                                                 strides_dim0: &[usize; 3],
                                                                 strides_dim1: &[usize; 3],
                                                                 iter_shape: &[usize; 2],
                                                                 mut output: Vec<T>,
                                                                 output_shape: Vec<usize>) -> Tensor<'b, T>
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op1 = operand1.mut_ptr();
        let op2 = operand2.mut_ptr();
        let dst = output.as_mut_ptr();
        
        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        
        for i in 0..iter_shape[1] {
            let ptr1 = op1.add(i * strides_dim1[0]);
            let ptr2 = op2.add(i * strides_dim1[1]);
            let dst = dst.add(i * strides_dim1[2]);

            sum_of_products(&[ptr1, ptr2, dst], &strides_dim0, iter_shape[0]);
        }

        Tensor::from_contiguous_owned_buffer(output_shape, output)
    }
}

pub(super) fn einsum_2operands_3labels<'b, T: SumOfProductsType>(operand1: &Tensor<T>,
                                                                 operand2: &Tensor<T>,
                                                                 strides_dim0: &[usize; 3],
                                                                 strides_dim1: &[usize; 3],
                                                                 strides_dim2: &[usize; 3],
                                                                 iter_shape: &[usize; 3],
                                                                 mut output: Vec<T>,
                                                                 output_shape: Vec<usize>) -> Tensor<'b, T>
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op1 = operand1.mut_ptr();
        let op2 = operand2.mut_ptr();
        let dst = output.as_mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);
        
        for i in 0..iter_shape[2] {
            for j in 0..iter_shape[1] {
                let ptr1 = op1.add(i * strides_dim2[0] + j * strides_dim1[0]);
                let ptr2 = op2.add(i * strides_dim2[1] + j * strides_dim1[1]);
                let dst = dst.add(i * strides_dim2[2] + j * strides_dim1[2]);

                sum_of_products(&[ptr1, ptr2, dst], &strides_dim0, iter_shape[0]);
            }
        }

        Tensor::from_contiguous_owned_buffer(output_shape, output)
    }
}
