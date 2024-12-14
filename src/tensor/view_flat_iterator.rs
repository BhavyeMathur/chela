use crate::dtype::RawDataType;
use crate::TensorView;

// interprets all contiguously stored dimensions as 1 big dimension
// if the entire array is stored contiguously, this results in just 1 long dimension
fn collapse_contiguous(shape: &Vec<usize>, stride: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {
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

#[non_exhaustive]
pub struct TensorViewFlatIter<T>
where
    T: RawDataType,
{
    ptr: *const T,
    shape: Vec<usize>,
    stride: Vec<usize>,
    ndims: usize,

    size: usize,
    iterator_index: usize,

    indices: Vec<usize>, // current index along each dimension
    flat_index: usize,
}

impl<T: RawDataType> TensorView<T> {
    pub fn flat_iter(&self) -> TensorViewFlatIter<T> {
        TensorViewFlatIter::from(&self)
    }
}

impl<T: RawDataType> TensorViewFlatIter<T> {
    fn from(tensor: &TensorView<T>) -> Self {
        let (shape, stride) = collapse_contiguous(&tensor.shape, &tensor.stride);
        let ndims = shape.len();

        Self {
            ptr: tensor.data.ptr.as_ptr().cast_const(),
            shape,
            stride,
            ndims,
            size: tensor.size(),
            iterator_index: 0,
            indices: vec![0; tensor.ndims],
            flat_index: 0,
        }
    }
}

impl<T> Iterator for TensorViewFlatIter<T>
where
    T: RawDataType,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_index == self.size {
            return None;
        }

        let current_index = self.flat_index as isize;

        for i in (0..self.ndims).rev() {
            self.indices[i] += 1;

            if self.indices[i] < self.shape[i] {
                self.flat_index += self.stride[i];
                break;
            }

            self.flat_index -= self.stride[i] * (self.shape[i] - 1);
            self.indices[i] = 0; // reset this dimension and carry over to the next
        }

        self.iterator_index += 1;
        Some(unsafe { *self.ptr.offset(current_index) })
    }
}

#[cfg(test)]
mod tests {
    use crate::view_flat_iterator::collapse_contiguous;
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
