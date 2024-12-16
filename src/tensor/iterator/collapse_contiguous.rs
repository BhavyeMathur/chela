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

#[cfg(test)]
mod tests {
    use super::collapse_contiguous;
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
}
