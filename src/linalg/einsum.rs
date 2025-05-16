use std::hint::assert_unchecked;
use crate::dtype::{NumericDataType, RawDataType};
use crate::iterator::multi_flat_index_generator::MultiFlatIndexGenerator;
use crate::tensor::MAX_DIMS;
use crate::{Tensor, TensorMethods};
use crate::linalg::specialized_einsum::*;

const MAX_EINSUM_OPERANDS: usize = 32;

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
fn parse_operand_subscripts<A: TensorMethods>(subscripts: &str,
                                              operand: &A,
                                              result: &mut [i8; MAX_DIMS],
                                              label_counts: &mut [u32; 128],
                                              label_dims: &mut [usize; 128],
                                              broadcast_dims: &mut usize) {
    if !subscripts.is_ascii() {
        panic!("einsum subscripts must be ascii");
    }
    let subscripts = subscripts.as_bytes();

    *broadcast_dims = operand.ndims() - subscripts.iter().filter(|c| c.is_ascii_alphanumeric()).count();

    let mut first_occurrence = [(MAX_DIMS + 3) as i8; 128];
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
            let axis = if ellipsis_index == -1 { i } else { (i - 2) + *broadcast_dims };
            if axis >= operand.ndims() {
                panic!("too many labels in einsum subscripts string");
            }

            if first_occurrence[label as usize] == (MAX_DIMS + 3) as i8 {
                first_occurrence[label as usize] = i as i8;

                result[axis] = label as i8;
            } else {
                result[axis] = first_occurrence[label as usize] - axis as i8;
            }

            label_counts[label as usize] += 1;
            if label_counts[label as usize] == 1 {
                label_dims[label as usize] = operand.shape()[i];
            } else if label_dims[label as usize] != operand.shape()[i] {
                panic!("the dimensions of axes corresponding to the same einsum label must match");
            }
        } else if label == b'.' {
            if ellipsis_index != -1 {
                panic!("einsum string may only contain single '..' for broadcasting");
            }
            check_ellipsis = true;
            ellipsis_index = i as isize;

            for j in i..i + *broadcast_dims {
                result[j] = 0;
            }
        } else if label != b' ' {
            panic!("invalid label '{}' in einsum string, subscripts must be letters", label);
        }
    }

    if check_ellipsis {
        panic!("einsum subscripts string contains '.' that is not part of an ellipsis ('..')")
    }

    if ellipsis_index == -1 && *broadcast_dims != 0 {
        panic!("too few labels in einsum subscripts string");
    }
}

/*
Same as parse_operand_subscripts() but output labels cannot be repeated
so result only has non-negative entries
 */
fn parse_output_subscripts(subscripts: &str,
                           result: &mut [i8; MAX_DIMS],
                           broadcast_dims: usize,
                           label_counts: &[u32; 128]) -> usize {
    if !subscripts.is_ascii() {
        panic!("einsum subscripts must be ascii");
    }
    let subscripts = subscripts.as_bytes();

    let mut found_ellipsis = false;
    let mut check_ellipsis = false;

    for (i, &label) in subscripts.iter().enumerate() {
        if check_ellipsis {
            check_ellipsis = false;
            if label != b'.' {
                panic!("einsum subscripts string contains '.' that is not part of an ellipsis ('..')");
            }
            continue;
        }

        if label.is_ascii_alphabetic() {
            if label_counts[label as usize] == 0 {
                panic!("einsum subscripts string included output subscript '{}' which never appeared in an input", label);
            }
            if subscripts.split_at(i + 1).1.contains(&label) {
                panic!("einsum subscripts includes output label '{}' multiple times", label);
            }

            let axis = if found_ellipsis { (i - 2) + broadcast_dims } else { i };
            if axis > MAX_DIMS {
                panic!("too many labels in einsum subscripts string");
            }

            result[axis] = label as i8;
        } else if label == b'.' {
            if found_ellipsis {
                panic!("einsum string may only contain single '..' for broadcasting");
            }
            if i + broadcast_dims > MAX_DIMS {
                panic!("too many labels in einsum subscripts string");
            }

            for j in i..i + broadcast_dims {
                result[j] = 0;
            }
            check_ellipsis = true;
            found_ellipsis = true;
        } else if label != b' ' {
            panic!("invalid label '{}' in einsum string, subscripts must be letters", label);
        }
    }

    let ndims = subscripts.len();
    if found_ellipsis { (ndims - 2) + broadcast_dims } else { ndims }
}


fn reshape_shape_and_stride_for_einsum<'a, A: TensorMethods>(operand: &'a A,
                                                             labels: &[i8; MAX_DIMS],
                                                             output_dims: usize,
                                                             output_labels: &[i8]) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut new_stride = vec![0; output_dims];
    let mut new_shape = vec![0; output_dims];

    for idim in 0..operand.ndims() {
        let mut label = labels[idim];
        if label < 0 {
            label = labels[(idim as i8 + label) as usize];
        }

        if label == 0 {
            panic!("broadcasting in einsum is currently unsupported");
        } else {
            match output_labels.iter().position(|&val| val == label) {
                None => { return None; },
                Some(axis_in_output) => {
                    new_shape[axis_in_output] = operand.shape()[idim];
                    new_stride[axis_in_output] += operand.stride()[idim];
                }
            }
        }
    }

    Some((new_stride, new_shape))
}

fn try_reshape_for_einsum<'a, T: RawDataType>(operand: &'a Tensor<T>,
                                              labels: &[i8; MAX_DIMS],
                                              output_dims: usize,
                                              output_labels: &[i8]) -> Option<Tensor<'a, T>> {
    match reshape_shape_and_stride_for_einsum(operand, labels, output_dims, output_labels) {
        None => None,
        Some((new_stride, new_shape)) => {
            unsafe { Some(operand.reshaped_view(new_shape, new_stride)) }
        }
    }
}


/*
Collapses dimensions with repeated subscripts. For example in ii-> (trace) or ii->i (diagonal)
 */
fn reshape_operand_for_einsum<'a, T: RawDataType>(operand: &'a Tensor<'a, T>,
                                                  labels: &mut [i8; MAX_DIMS]) -> Tensor<'a, T> {
    // fast path if operand dimensions cannot be combined
    if labels.iter().all(|&val| val >= 0) {
        return operand.view();
    }

    let mut new_stride = Vec::with_capacity(operand.ndims());
    let mut new_shape = Vec::with_capacity(operand.ndims());
    let mut icombinemap = [0; MAX_DIMS];
    let mut icombine = 0;

    for axis in 0..operand.ndims() {
        let label = labels[axis];
        let dimension = operand.shape()[axis];
        let stride = operand.stride()[axis];

        if label >= 0 { // label seen for the first time
            icombinemap[axis] = icombine;
            icombine += 1;

            new_shape.push(dimension);
            new_stride.push(stride);
        } else { // repeated label
            let i = icombinemap[(axis as i8 + label) as usize];
            icombinemap[axis] = MAX_DIMS + 1;
            new_stride[i] += stride;

            if new_shape[i] != dimension {
                panic!("dimensions in operand for collapsing don't match")
            }
        }
    }

    for axis in 0..operand.ndims() {
        let i = icombinemap[axis];
        if i != MAX_DIMS + 1 {
            labels[i] = labels[axis];
        }
    }

    unsafe { Tensor::reshaped_view(&operand, new_shape, new_stride) }
}

fn operand_stride_for_einsum<A: TensorMethods>(operand: &A,
                                               operand_labels: &[i8; MAX_DIMS],
                                               result: &mut [usize],
                                               iter_labels: &[i8]) {
    for (i, &label) in iter_labels.iter().enumerate() {
        if label == 0 {
            panic!("broadcasting in einsum is currently unsupported");
        } else {
            for (index, &op_label) in operand_labels.iter().enumerate() {
                if index == operand.ndims() {
                    break;
                }

                let op_label =
                    if op_label >= 0 { op_label } else { operand_labels[(index as i8 + op_label) as usize] };

                if label == op_label {
                    result[i] += operand.stride()[index];
                }
            }
        }
    }
}

pub fn einsum_view<'b, T: NumericDataType>(operand: &'b Tensor<T>,
                                           subscripts: (&str, &str)) -> Option<Tensor<'b, T>> {
    let mut labels = [0; MAX_DIMS];
    let mut output_labels = [0; MAX_DIMS];
    let mut label_counts = [0; 128];

    let mut broadcast_dims = 0;
    let mut max_broadcast_dims = 0;

    parse_operand_subscripts(subscripts.0, operand, &mut labels, &mut label_counts, &mut [0; 128], &mut broadcast_dims);
    max_broadcast_dims = max_broadcast_dims.max(broadcast_dims);

    let output_dims = parse_output_subscripts(subscripts.1, &mut output_labels, max_broadcast_dims, &label_counts);
    let output_labels = &output_labels[0..output_dims];

    // try returning a reshaped view of the tensor
    try_reshape_for_einsum(operand, &labels, output_dims, output_labels)
}

// TODO this is around 2.5x slower than NumPy. We need to speed it up!
pub fn einsum<'b, const N: usize, T: NumericDataType>(operands: &[&Tensor<T>; N],
                                                      subscripts: ([&str; N], &str)) -> Tensor<'b, T> {
    if N == 0 {
        return Tensor::scalar(T::one());
    }

    let mut operand_labels = [[0; MAX_DIMS]; N];
    let mut output_labels = [0; MAX_DIMS];
    let mut label_counts = [0; 128];
    let mut label_dims = [0; 128];

    let mut broadcast_dims = 0;
    let mut max_broadcast_dims = 0;


    // parse input & output subscripts

    for (i, (subscript, &operand)) in subscripts.0.iter().zip(operands.iter()).enumerate() {
        parse_operand_subscripts(subscript, operand, &mut operand_labels[i], &mut label_counts, &mut label_dims, &mut broadcast_dims);
        max_broadcast_dims = max_broadcast_dims.max(broadcast_dims);
    }

    let output_dims = parse_output_subscripts(subscripts.1, &mut output_labels, max_broadcast_dims, &label_counts);

    let mut iter_ndims = output_dims;
    let mut iter_labels = output_labels;
    let output_labels = &output_labels[0..output_dims];


    // check if the output can be generated by only reshaping the tensor

    if N == 1 && reshape_shape_and_stride_for_einsum(operands[0], &operand_labels[0], output_dims, output_labels).is_some() {
        eprintln!("\x1b[33mchela warning: use einsum_view() or reshape() to improve performance for this operation\x1b[0m");
    }


    // process input operands and combine dimensions with duplicate subscripts

    let mut reshaped_operands = Vec::with_capacity(N);
    for (operand, labels) in operands.iter().zip(operand_labels.iter_mut()) {
        reshaped_operands.push(reshape_operand_for_einsum(operand, labels));
    }


    // set up the output buffer after calculating its shape

    let mut output_shape = Vec::with_capacity(output_dims);
    for &label in output_labels.iter() {
        output_shape.push(label_dims[label as usize]);
    }
    let output_len = output_shape.iter().product();
    let mut output = vec![T::zero(); output_len];


    // create iterators to iterate over the operands in the correct order

    let mut iter_shape = output_shape.clone();
    let mut nterms = 1;
    for label in 0i8..=127 {
        if label_counts[label as usize] == 0 || output_labels.contains(&label) {
            continue;
        }
        if iter_ndims >= MAX_DIMS {
            panic!("too many subscripts in einsum");
        }

        let dimension = label_dims[label as usize];
        nterms *= dimension;

        iter_labels[iter_ndims] = label;
        iter_shape.push(dimension);
        iter_ndims += 1;
    }
    let iter_labels = &iter_labels[0..iter_ndims];


    // create iterator to traverse operand values in the correct order

    let mut operand_strides = [[0; MAX_DIMS]; N];
    for ((operand, labels), stride) in reshaped_operands.iter()
                                                        .zip(operand_labels.iter_mut())
                                                        .zip(operand_strides.iter_mut()) {
        operand_stride_for_einsum(operand, labels, stride, iter_labels)
    }


    // accelerated loops for specific structures

    if N == 2 && iter_ndims == 3 {
        return einsum_2operands_3labels(operands[0], operands[1],
                                        <&[usize; 3]>::try_from(&operand_strides[0][0..3]).unwrap(),
                                        <&[usize; 3]>::try_from(&operand_strides[1][0..3]).unwrap(),
                                        <&[usize; 3]>::try_from(&iter_shape[0..3]).unwrap(),
                                        output, output_shape);
    }


    // main einsum calculation loop

    let mut operand_indices = MultiFlatIndexGenerator::from(&iter_shape, &operand_strides);

    for dst in output.iter_mut() {
        let mut sum = T::zero();

        let mut i = nterms;
        while i != 0 {
            i -= 1;

            unsafe {
                let indices = operand_indices.cur_indices();

                sum += indices.iter()
                              .zip(operands.iter())
                              .map(|(&i, operand)|
                                  *operand.ptr.add(i).as_ptr())
                              .product();

                operand_indices.increment_flat_indices();
            }
        }
        *dst = sum;
    }
    unsafe { Tensor::from_contiguous_owned_buffer(output_shape, output) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_invalid_ellipsis_multiple() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 4]);
        let mut result = [3; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("a..b..", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
    }

    #[test]
    #[should_panic]
    fn test_invalid_ellipsis() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 6]);
        let mut result = [1; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("a.b", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
    }

    #[test]
    #[should_panic]
    fn test_invalid_ellipsis_at_end() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 3]);
        let mut result = [4; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("ab.", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
    }

    #[test]
    fn test_unique_labels2() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 6]);
        let mut result = [1; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 1;

        parse_operand_subscripts("abcdef", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..6], [97, 98, 99, 100, 101, 102]);
        assert_eq!(broadcast_dims, 0);
    }

    #[test]
    fn test_parse_operand_subscripts_with_repeats() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 6]);
        let mut result = [5; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 60;

        parse_operand_subscripts("abbcbc", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);

        assert_eq!(result[0..6], [97, 98, -1, 99, -3, -2]);
        assert_eq!(broadcast_dims, 0);
    }

    #[test]
    fn test_ellipsis_middle() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 7]);
        let mut result = [9; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("ab..bc", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..7], [97, 98, 0, 0, 0, -4, 99]);
        assert_eq!(broadcast_dims, 3);
    }

    #[test]
    fn test_ellipsis_middle2() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 9]);
        let mut result = [9; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("ab..bc", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..9], [97, 98, 0, 0, 0, 0, 0, -6, 99]);
        assert_eq!(broadcast_dims, 5);
    }

    #[test]
    fn test_ellipsis_start() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 5]);
        let mut result = [2; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("..ab", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..5], [0, 0, 0, 97, 98]);
        assert_eq!(broadcast_dims, 3);
    }

    #[test]
    fn test_multiple_operands() {
        let tensor: Tensor<f32> = Tensor::zeros([1; 4]);
        let mut result = [0; MAX_DIMS];
        let mut label_counts = [0; 128];
        let mut label_dims = [0; 128];
        let mut broadcast_dims = 0;

        parse_operand_subscripts("abb..", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..4], [97, 98, -1, 0]);
        assert_eq!(broadcast_dims, 1);

        let tensor: Tensor<f32> = Tensor::zeros([1; 5]);
        parse_operand_subscripts("abb..", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..5], [97, 98, -1, 0, 0]);
        assert_eq!(broadcast_dims, 2);

        let tensor: Tensor<f32> = Tensor::zeros([1; 4]);
        parse_operand_subscripts("baba", &tensor, &mut result, &mut label_counts, &mut label_dims, &mut broadcast_dims);
        assert_eq!(result[0..4], [98, 97, -2, -2]);
        assert_eq!(broadcast_dims, 0);
    }
}
