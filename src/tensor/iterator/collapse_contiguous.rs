// interprets all contiguously stored dimensions as 1 big dimension
// if the entire array is stored contiguously, this results in just 1 long dimension
pub(super) fn collapse_contiguous(shape: &Vec<usize>, stride: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
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
        return (shape.clone(), stride.clone());
    }

    let mut collapsed_shape = shape[..ndims].to_vec();
    let mut collapsed_stride = stride[..ndims].to_vec();

    collapsed_shape.push(stride_if_contiguous);
    collapsed_stride.push(1);

    (collapsed_shape, collapsed_stride)
}
