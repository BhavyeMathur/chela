use crate::iterator::multi_flat_index_generator::MultiFlatIndexGenerator;
use crate::linalg::sum_of_products::*;
use crate::ndarray::{MAX_ARGS, MAX_DIMS};
use crate::{first_n_elements, NdArray};
use std::hint::assert_unchecked;
use std::ptr::null_mut;

pub(super) unsafe fn unspecialized_einsum_loop<T: SumOfProductsType>(operands: &[&NdArray<T>],
                                                                     strides: &[[usize; MAX_ARGS]; MAX_DIMS],
                                                                     iter_ndims: usize,
                                                                     iter_shape: &[usize],
                                                                     dst: *mut T) {
    let n_operands = operands.len();

    let strides = &strides[0..iter_ndims];
    let inner_stride = &strides[0][..n_operands + 1];
    let mut indices_iter = MultiFlatIndexGenerator::from(n_operands + 1, &iter_shape[1..], &strides[1..]);

    let sum_of_products = get_sum_of_products_function_generic_nops(inner_stride);

    let mut base_ptrs = [null_mut(); MAX_ARGS];
    let mut ptrs = base_ptrs;
    let ptrs = &mut ptrs[0..n_operands + 1];

    base_ptrs[n_operands] = dst;

    unsafe {
        for (i, &operand) in operands.iter().enumerate() {
            base_ptrs[i] = operand.mut_ptr();
        }

        for _ in 0..iter_shape[1..].iter().product() {
            let indices = indices_iter.cur_indices();

            for (i, &index) in indices[..n_operands + 1].iter().enumerate() {
                ptrs[i] = base_ptrs[i].add(index);
            }

            sum_of_products(ptrs, inner_stride, iter_shape[0]);
            indices_iter.increment_flat_indices();
        }
    }
}

pub(super) unsafe fn try_specialized_einsum_loop<T: SumOfProductsType>(operands: &[&NdArray<T>],
                                                                       strides: &[[usize; MAX_ARGS]; MAX_DIMS],
                                                                       iter_ndims: usize,
                                                                       iter_shape: &[usize],
                                                                       dst: *mut T) -> bool {
    let n_operands = operands.len();

    if n_operands == 1 {
        if iter_ndims == 2 {
            einsum_1operand_2labels(operands[0],
                                    first_n_elements!(strides[0], 2),
                                    first_n_elements!(strides[1], 2),
                                    first_n_elements!(iter_shape, 2),
                                    dst);
        } else if iter_ndims == 3 {
            einsum_1operand_3labels(operands[0],
                                    first_n_elements!(strides[0], 2),
                                    first_n_elements!(strides[1], 2),
                                    first_n_elements!(strides[2], 2),
                                    first_n_elements!(iter_shape, 3),
                                    dst);
        }
    } else if n_operands == 2 {
        if iter_ndims == 2 {
            einsum_2operands_2labels(operands[0], operands[1],
                                     first_n_elements!(strides[0], 3),
                                     first_n_elements!(strides[1], 3),
                                     first_n_elements!(iter_shape, 2),
                                     dst);
        } else if iter_ndims == 3 {
            einsum_2operands_3labels(operands[0], operands[1],
                                     first_n_elements!(strides[0], 3),
                                     first_n_elements!(strides[1], 3),
                                     first_n_elements!(strides[2], 3),
                                     first_n_elements!(iter_shape, 3),
                                     dst);
        }
    }

    (n_operands == 1 || n_operands == 2) && (iter_ndims == 2 || iter_ndims == 3)
}


pub(super) unsafe fn einsum_1operand_2labels<T: SumOfProductsType>(operand: &NdArray<T>,
                                                                   strides_dim0: &[usize; 2],
                                                                   strides_dim1: &[usize; 2],
                                                                   iter_shape: &[usize; 2],
                                                                   dst: *mut T)
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op = operand.mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);

        for i in 0..iter_shape[1] {
            let src = op.add(i * strides_dim1[0]);
            let dst = dst.add(i * strides_dim1[1]);

            sum_of_products(&[src, dst], strides_dim0, iter_shape[0]);
        }
    }
}

pub(super) unsafe fn einsum_1operand_3labels<T: SumOfProductsType>(operand: &NdArray<T>,
                                                                   strides_dim0: &[usize; 2],
                                                                   strides_dim1: &[usize; 2],
                                                                   strides_dim2: &[usize; 2],
                                                                   iter_shape: &[usize; 3],
                                                                   dst: *mut T)
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op = operand.mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);

        for i in 0..iter_shape[2] {
            for j in 0..iter_shape[1] {
                let src = op.add(i * strides_dim2[0] + j * strides_dim1[0]);
                let dst = dst.add(i * strides_dim2[1] + j * strides_dim1[1]);

                sum_of_products(&[src, dst], strides_dim0, iter_shape[0]);
            }
        }
    }
}

pub(super) unsafe fn einsum_2operands_2labels<T: SumOfProductsType>(operand1: &NdArray<T>,
                                                                    operand2: &NdArray<T>,
                                                                    strides_dim0: &[usize; 3],
                                                                    strides_dim1: &[usize; 3],
                                                                    iter_shape: &[usize; 2],
                                                                    dst: *mut T)
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op1 = operand1.mut_ptr();
        let op2 = operand2.mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);

        for i in 0..iter_shape[1] {
            let ptr1 = op1.add(i * strides_dim1[0]);
            let ptr2 = op2.add(i * strides_dim1[1]);
            let dst = dst.add(i * strides_dim1[2]);

            sum_of_products(&[ptr1, ptr2, dst], strides_dim0, iter_shape[0]);
        }
    }
}

pub(super) unsafe fn einsum_2operands_3labels<T: SumOfProductsType>(operand1: &NdArray<T>,
                                                                    operand2: &NdArray<T>,
                                                                    strides_dim0: &[usize; 3],
                                                                    strides_dim1: &[usize; 3],
                                                                    strides_dim2: &[usize; 3],
                                                                    iter_shape: &[usize; 3],
                                                                    dst: *mut T)
{
    let sum_of_products = get_sum_of_products_function(strides_dim0);

    unsafe {
        let op1 = operand1.mut_ptr();
        let op2 = operand2.mut_ptr();

        assert_unchecked(iter_shape[0] > 0);
        assert_unchecked(iter_shape[1] > 0);
        assert_unchecked(iter_shape[2] > 0);

        for i in 0..iter_shape[2] {
            for j in 0..iter_shape[1] {
                let ptr1 = op1.add(i * strides_dim2[0] + j * strides_dim1[0]);
                let ptr2 = op2.add(i * strides_dim2[1] + j * strides_dim1[1]);
                let dst = dst.add(i * strides_dim2[2] + j * strides_dim1[2]);

                sum_of_products(&[ptr1, ptr2, dst], strides_dim0, iter_shape[0]);
            }
        }
    }
}
