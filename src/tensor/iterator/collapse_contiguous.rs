// interprets all contiguously stored dimensions as 1 big dimension
// if the entire array is stored contiguously, this results in just 1 long dimension
pub(in crate::tensor) fn collapse_contiguous(shape: &[usize], stride: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if stride.last() != Some(&1) {
        return (shape.to_vec(), stride.to_vec());
    }

    let mut stride_if_contiguous = 1;
    let mut ndims = shape.len();

    for (&axis_length, &actual_stride) in shape.iter().zip(stride.iter()).rev() {
        if stride_if_contiguous != actual_stride {
            break;
        }
        ndims -= 1;
        stride_if_contiguous *= axis_length;
    }

    if stride_if_contiguous == 1 { // none of the dimensions are contiguous
        return (shape.to_vec(), stride.to_vec());
    }

    let mut collapsed_shape = shape[..ndims].to_vec();
    let mut collapsed_stride = stride[..ndims].to_vec();

    collapsed_shape.push(stride_if_contiguous);
    collapsed_stride.push(1);

    (collapsed_shape, collapsed_stride)
}

pub(in crate::tensor) fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    let mut stride_if_contiguous = 1;

    for (&axis_length, &actual_stride) in shape.iter().zip(stride.iter()).rev() {
        if stride_if_contiguous != actual_stride {
            return false;
        }
        stride_if_contiguous *= axis_length;
    }

    true
}

// Examples
//
// shape (2, 3), stride (3, 1) -> shape (6,), stride (1,)
// [[0, 1, 2], [3, 4, 5]] -> [0, 1, 2, 3, 4, 5]
//
// shape (2, 3), stride (6, 2) -> shape (6,), stride (2,)
// [[0, 2, 4], [6, 8, 10]] -> [0, 2, 4, 6, 8, 10]
//
// shape (2, 3), stride (5, 2) -> shape (2, 3), stride (5, 2)
// [[0, 2, 4], [5, 7, 9]] -> [[0, 2, 4], [5, 7, 9]]
//
// shape (2, 2, 2), stride (6, 3, 2) -> shape (4, 2), stride (3, 2)
// [[[0, 2], [3, 5]], [[6, 8], [9, 11]]] -> [[0, 2], [3, 5], [6, 8], [9, 11]]
pub(in crate::tensor) fn collapse_to_uniform_stride(shape: &[usize], stride: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let ndims = shape.len();
    if ndims == 0 {
        return (vec![], vec![]);
    }

    let mut new_shape = Vec::with_capacity(ndims);
    let mut new_stride = Vec::with_capacity(ndims);

    new_shape.push(shape[0]);
    new_stride.push(stride[0]);

    let mut last_idx = 0;

    for i in 1..ndims {
        // check if this dimension can be collapsed into the previous one
        if new_stride[last_idx] == shape[i] * stride[i] {
            new_shape[last_idx] *= shape[i];  // collapse by merging dimension into the previous one
            new_stride[last_idx] = stride[i];
        } else {
            new_shape.push(shape[i]);  // otherwise, start a new dimension
            new_stride.push(stride[i]);
            last_idx += 1;
        }
    }

    (new_shape, new_stride)
}


pub(in crate::tensor) fn has_uniform_stride(shape: &[usize], stride: &[usize]) -> Option<usize> {
    let (_, new_stride) = collapse_to_uniform_stride(shape, stride);

    // TODO don't need to calculate entire collapsed stride so this can be faster
    if new_stride.len() == 1 {
        return Some(new_stride[0]);
    }
    None
}


#[cfg(test)]
mod tests {
    use super::collapse_contiguous;
    use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
    use crate::{s, Tensor};

    #[test]
    fn test_collapse_contiguous() {
        let a = Tensor::from([
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[12, 13, 14], [15, 16, 17]],
        ]);

        let (shape, stride) = collapse_contiguous(&a.shape, &a.stride);
        assert_eq!(shape, [18]);
        assert_eq!(stride, [1]);

        let b = a.slice(s![.., 0]);
        let (shape, stride) = collapse_contiguous(&b.shape, &b.stride);
        assert_eq!(shape, [3, 3]);
        assert_eq!(stride, [6, 1]);

        let b = a.slice(s![1]);
        let (shape, stride) = collapse_contiguous(&b.shape, &b.stride);
        assert_eq!(shape, [6]);
        assert_eq!(stride, [1]);

        let b = a.slice(s![..2, 1, 1..]);
        let (shape, stride) = collapse_contiguous(&b.shape, &b.stride);
        assert_eq!(shape, [2, 2]);
        assert_eq!(stride, [6, 1]);
    }

    // courtesy of ChatGPT
    #[test]
    fn test_collapse_to_uniform_stride() {
        // Example 1
        let shape = [2, 3];
        let stride = [3, 1];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [6]);
        assert_eq!(b, [1]); // Collapsed stride should match the inner-most dimension's stride.

        // Example 2
        let shape = [2, 3];
        let stride = [6, 2];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [6]);
        assert_eq!(b, [2]); // Collapsed as strides are consistent.

        // Example 3
        let shape = [2, 3];
        let stride = [5, 2];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [2, 3]);
        assert_eq!(b, [5, 2]); // Cannot collapse due to inconsistent strides.

        // Example 4
        let shape = [2, 2, 2];
        let stride = [6, 3, 2];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [4, 2]);
        assert_eq!(b, [3, 2]); // Collapsed outer two dimensions.

        // Additional Example 1
        let shape = [3, 4, 5];
        let stride = [20, 5, 1];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [60]);
        assert_eq!(b, [1]); // Fully collapsed due to consistent strides.

        // Additional Example 2
        let shape = [4, 5, 6];
        let stride = [30, 6, 1];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [120]);
        assert_eq!(b, [1]); // Fully collapsed due to consistent strides.

        // Additional Example 3
        let shape = [3, 3, 3];
        let stride = [9, 3, 1];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [27]);
        assert_eq!(b, [1]); // Fully collapsed into a single dimension.

        // Edge Case: Empty shape and stride
        let shape = [];
        let stride = [];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, []);
        assert_eq!(b, []); // Should handle empty inputs correctly.

        // Edge Case: Single dimension
        let shape = [10];
        let stride = [1];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [10]);
        assert_eq!(b, [1]); // Single dimension remains unchanged.

        // Edge Case: Non-contiguous strides
        let shape = [2, 3];
        let stride = [4, 2];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [2, 3]);
        assert_eq!(b, [4, 2]); // Cannot collapse due to non-contiguous strides.

        // Edge Case: Zero stride
        let shape = [3, 3, 3];
        let stride = [0, 1, 0];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [3, 3, 3]);
        assert_eq!(b, [0, 1, 0]);

        // Edge Case: Zero stride with some dimensions collapsed
        let shape = [5, 2, 3, 3, 4, 3];
        let stride = [6, 3, 0, 4, 1, 0];
        let (a, b) = collapse_to_uniform_stride(&shape, &stride);
        assert_eq!(a, [10, 3, 12, 3]);
        assert_eq!(b, [3, 0, 1, 0]);
    }
}
