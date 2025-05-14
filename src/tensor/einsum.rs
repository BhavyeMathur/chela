use crate::buffer_iterator::BufferIterator;
use crate::dtype::NumericDataType;
use crate::Tensor;
use std::collections::HashMap;

// Let A be a tensor with shape (I, J)
// Let B be a tensor with shape (J, K)
// Suppose our subscripts describe "ij,jk->ik"
//
// Then subscript_to_index will be {i: 0, j: 1, k: 2}
// and axes_lengths will be [I, J, K]
fn parse_subscripts<const N: usize, T: NumericDataType>(operands: &[&Tensor<T>; N], subscripts: &[&str; N])
                                                        -> (HashMap<char, usize>, Vec<usize>) {
    let mut subscript_to_index = HashMap::new();
    let mut axes_lengths = Vec::new();

    for (tensor, subscripts) in operands.iter().zip(subscripts) {
        assert_eq!(subscripts.len(), tensor.ndims(), "einstein sum subscripts must have same length as dimensions for tensor {}", tensor.ndims());

        for (&axis_length, subscript) in tensor.shape().iter().zip(subscripts.chars()) {
            match subscript_to_index.get(&subscript) {
                None => {
                    subscript_to_index.insert(subscript, subscript_to_index.len());
                    axes_lengths.push(axis_length);
                }
                Some(&i) => { assert_eq!(axes_lengths[i], axis_length, "the lengths of axes corresponding to the same index must match"); }
            };
        }
    }

    (subscript_to_index, axes_lengths)
}

// Suppose subscript_to_index is {i: 0, j: 1, k: 2} and subscripts is 'ik'
// and suppose stride is (K, 1)
// Then this function should output [K, 0, 1]
fn get_augmented_stride(subscripts: &str, stride: &[usize], subscript_to_index: &HashMap<char, usize>) -> Vec<usize> {
    let mut result = vec![0; subscript_to_index.len()];
    for (&stride, subscript) in stride.iter().zip(subscripts.chars()) {
        let index = subscript_to_index[&subscript];
        result[index] += stride;
    }
    result
}

fn iter_operands_for_einsum<const N: usize, T: NumericDataType>(operands: [&Tensor<T>; N],
                                                                subscripts: &[&str; N],
                                                                axes_lengths: &Vec<usize>,
                                                                subscript_to_index: &HashMap<char, usize>) -> Vec<BufferIterator<T>> {
    let mut result = Vec::with_capacity(subscripts.len());

    for (tensor, subscript) in operands.iter().zip(subscripts) {
        let stride = get_augmented_stride(subscript, tensor.stride(), &subscript_to_index);
        result.push(unsafe {
            BufferIterator::from_reshaped_view(&tensor, &axes_lengths, &stride)
        });
    }

    result
}

pub fn einsum<'b, const N: usize, T: NumericDataType>(operands: [&Tensor<T>; N], subscripts: ([&str; N], &str)) -> Tensor<'b, T> {
    let (input_subscripts, result_subscripts) = subscripts;
    let (subscript_to_index, axes_lengths) = parse_subscripts(&operands, &input_subscripts);

    let mut result_shape = Vec::with_capacity(result_subscripts.len());
    let mut output_subscript_indices = Vec::with_capacity(result_subscripts.len());

    for subscript in result_subscripts.chars() {
        match subscript_to_index.get(&subscript) {
            Some(&index) => {
                result_shape.push(axes_lengths[index]);
                output_subscript_indices.push(index);
            }
            None => { panic!("einstein sum subscripts string included output subscript '{}' which never appeared in an input", subscript) }
        }
    }

    let len = result_shape.iter().product();
    let result = unsafe {
        Tensor::from_contiguous_owned_buffer(result_shape, vec![T::zero(); len])
    };

    let result_iter = unsafe {
        BufferIterator::from_reshaped_view(&result, &axes_lengths, &get_augmented_stride(result_subscripts, result.stride(), &subscript_to_index))
    };

    let mut input_iters = iter_operands_for_einsum(operands, &input_subscripts, &axes_lengths, &subscript_to_index);

    for dst in result_iter {
        let mut product = T::one();
        for src in input_iters.iter_mut() {
            product *= unsafe { *src.next().unwrap() };
        }

        unsafe { *dst += product };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_augmented_stride() {
        let index = HashMap::from([('i', 0), ('j', 1), ('k', 2)]);

        let stride = vec![10, 1];
        let subscripts = "ik";
        assert_eq!(get_augmented_stride(&subscripts, &stride, &index), vec![10, 0, 1]);

        let stride = vec![10, 1];
        let subscripts = "ij";
        assert_eq!(get_augmented_stride(&subscripts, &stride, &index), vec![10, 1, 0]);

        let stride = vec![30, 1];
        let subscripts = "jk";
        assert_eq!(get_augmented_stride(&subscripts, &stride, &index), vec![0, 30, 1]);
    }
}
