use crate::buffer_iterator::BufferIterator;
use crate::dtype::NumericDataType;
use crate::Tensor;
use std::collections::HashMap;
/*
See NumPy's implementation of this function for more details.

Returns an array with 'ndims' labels after parsing the subscripts for one operand with
    - the ASCII code of the label on its first occurrence
    - the (negative) offset to the first occurrence of the label for repeated labels
    - zero for broadcast dimensions (if the subscript has ellipsis)

Examples:
    subscripts="abbcbc",  ndim=6 -> result = [97, 98, -1, 99, -3, -2]
    subscripts="ab..bc", ndim=6 -> result = [97, 98, 0, 0, -3, 99]
 */
fn parse_operand_subscripts<const N: usize, T: NumericDataType>(subscripts: &str, operand: &Tensor<T>,
                                                                result: &mut [i32; N],
                                                                label_counts: &mut [u32; 128]) {
    if !subscripts.is_ascii() {
        panic!("einsum subscripts must be ascii");
    }
    if subscripts.len() > operand.ndims() {
        panic!("too many labels in einsum subscripts string");
    }
    let broadcast_dims = operand.ndims() - subscripts.len();
    let subscripts = subscripts.as_bytes();

    let mut first_occurrence = [0; 128];
    let mut ellipsis_index: isize = -1;
    let mut check_ellipsis = false;

    for (i, &label) in subscripts.iter().enumerate() {
        if check_ellipsis {
            check_ellipsis = false;
            if label != b'.' {
                panic!("einsum subscripts string contains '.' that is not part of an ellipsis ('..')")
            }
            continue;
        }

        if label.is_ascii_alphabetic() {
            let val;

            if label_counts[label as usize] == 0 {
                first_occurrence[label as usize] = i as i32;
                val = label as i32;
            } else {
                val = first_occurrence[label as usize] - i as i32;
            };
            label_counts[label as usize] += 1;

            if ellipsis_index == -1 {
                result[i] = val;
            } else {
                result[i + broadcast_dims] = val;
            }
        } else if label == b'.' {
            if ellipsis_index != -1 {
                panic!("einsum string may only contain single '#' for broadcasting");
            }
            check_ellipsis = true;
            ellipsis_index = i as isize;
            result[i] = 0;
            result[i + 1] = 0;
        } else if label != b' ' {
            panic!("invalid label '{}' in einsum string, subscripts must be letters", label);
        }
    }

    if check_ellipsis {
        panic!("einsum subscripts string contains '.' that is not part of an ellipsis ('..')")
    }

    if ellipsis_index == -1 && broadcast_dims != 0 {
        panic!("too few labels in einsum subscripts string");
    }
}

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

    for (tensor, subscripts) in operands.iter().zip(subscripts.iter()) {
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

    for (&tensor, &subscript) in operands.iter().zip(subscripts.iter()) {
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
            None => {
                panic!("einstein sum subscripts string included output subscript '{}' which never appeared in an input", subscript)
            }
        }
    }

    let len = result_shape.iter().product();
    let result = unsafe {
        Tensor::from_contiguous_owned_buffer(result_shape, vec![T::zero(); len])
    };

    let result_iter = unsafe {
        BufferIterator::from_reshaped_view(&result, &axes_lengths, &get_augmented_stride(result_subscripts, result.stride(), &subscript_to_index))
    };

    // note that every iterator in input_iters is guaranteed to be the same length as result_iter
    let mut input_iters = iter_operands_for_einsum(operands, &input_subscripts, &axes_lengths, &subscript_to_index);

    for dst in result_iter {
        let mut product = T::one();
        for src in input_iters.iter_mut() {
            product *= unsafe { *src.next().unwrap_unchecked() };
        }

        unsafe { *dst += product };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_invalid_ellipsis_multiple() {
        const N: usize = 4;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [0; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("a..b..", &tensor, &mut result, &mut label_counts);
    }

    #[test]
    #[should_panic]
    fn test_invalid_ellipsis() {
        const N: usize = 4;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [0; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("a.b", &tensor, &mut result, &mut label_counts);
    }

    #[test]
    #[should_panic]
    fn test_invalid_ellipsis_at_end() {
        const N: usize = 4;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [0; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("ab.", &tensor, &mut result, &mut label_counts);
    }

    #[test]
    fn test_unique_labels() {
        const N: usize = 6;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [0; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("abcdef", &tensor, &mut result, &mut label_counts);
        assert_eq!(result, [97, 98, 99, 100, 101, 102]);
    }

    #[test]
    fn test_parse_operand_subscripts_with_repeats() {
        const N: usize = 6;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [-45; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("abbcbc", &tensor, &mut result, &mut label_counts);

        assert_eq!(result, [97, 98, -1, 99, -3, -2]);
    }

    #[test]
    fn test_ellipsis_middle() {
        const N: usize = 7;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [0; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("ab..bc", &tensor, &mut result, &mut label_counts);
        assert_eq!(result, [97, 98, 0, 0, 0, -3, 99]);
    }

    #[test]
    fn test_ellipsis_start() {
        const N: usize = 5;
        let tensor: Tensor<f32> = Tensor::zeros([1; N]);
        let mut result = [0; N];
        let mut label_counts = [0; 128];

        parse_operand_subscripts("..ab", &tensor, &mut result, &mut label_counts);
        assert_eq!(result, [0, 0, 0, 97, 98]);
    }


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
