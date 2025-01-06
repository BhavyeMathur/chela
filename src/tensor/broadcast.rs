use crate::dtype::RawDataType;
use crate::traits::to_vec::ToVec;
use crate::Tensor;

fn pad_dimensions(shape: &[usize], stride: &[usize], ndims: usize) -> (Vec<usize>, Vec<usize>) {
    let mut new_shape = Vec::with_capacity(ndims);
    let mut new_stride = Vec::with_capacity(ndims);

    for _ in 0..(ndims - shape.len()) {
        new_shape.push(1);
        new_stride.push(0);
    }

    new_shape.extend(shape);
    new_stride.extend(stride);

    (new_shape, new_stride)
}

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub fn broadcast_to(&'a self, shape: impl ToVec<usize>) -> Tensor<'a, T> {
        let shape = shape.to_vec();
        let ndims = shape.len();
        assert!(ndims >= self.ndims(), "cannot broadcast to fewer dimensions");

        let (mut broadcast_shape, mut broadcast_stride) = pad_dimensions(&self.shape, &self.stride, ndims);

        for axis in 0..ndims {
            if shape[axis] == broadcast_shape[axis] {
                continue;
            }

            if broadcast_shape[axis] != 1 {
                panic!("tensor is not compatible with the desired shape");
            }

            broadcast_shape[axis] = shape[axis];
            broadcast_stride[axis] = 0;
        }

        unsafe { self.reshaped_view(broadcast_shape, broadcast_stride) }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast() {
        let tensor = Tensor::from([1, 2, 3]);
        let tensor = tensor.broadcast_to([3, 3]);
        assert_eq!(tensor.shape(), [3, 3]);
        assert_eq!(tensor, Tensor::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]));
    }

    #[test]
    fn test_broadcast_scalar_to_higher_dims() {
        let tensor = Tensor::from([42]);
        let tensor = tensor.broadcast_to([2, 3]);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(tensor, Tensor::from([[42, 42, 42], [42, 42, 42]]));
    }

    #[test]
    fn test_broadcast_matrix_to_higher_dims() {
        let tensor = Tensor::from([[1, 2], [3, 4]]);
        let tensor = tensor.broadcast_to([3, 2, 2]);
        assert_eq!(tensor.shape(), [3, 2, 2]);
        assert_eq!(tensor, Tensor::from([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]));
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_shapes() {
        let tensor = Tensor::from([1, 2, 3]);
        tensor.broadcast_to([3, 5]);
    }

    #[test]
    fn test_broadcast_identity() {
        let tensor = Tensor::from([1, 2, 3]);
        let tensor = tensor.broadcast_to([3]);
        assert_eq!(tensor.shape(), [3]);
        assert_eq!(tensor, Tensor::from([1, 2, 3]));
    }

    #[test]
    fn test_broadcast_unchanged() {
        let tensor = Tensor::from([1, 2, 3]);
        let tensor = tensor.broadcast_to([1, 3]);
        assert_eq!(tensor.shape(), [1, 3]);
        assert_eq!(tensor, Tensor::from([[1, 2, 3]]));
    }

    #[test]
    fn test_broadcast_high_dims() {
        let tensor = Tensor::from([[1], [2], [3]]);
        let tensor = tensor.broadcast_to([3, 3, 3]);
        assert_eq!(tensor.shape(), [3, 3, 3]);
        assert_eq!(tensor, Tensor::from(
            [[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        )
        );
    }

    #[test]
    fn test_broadcast_single_dimensional_expansion() {
        let tensor = Tensor::from([1, 2, 3]);
        let tensor = tensor.broadcast_to([3, 1, 3]);
        assert_eq!(tensor.shape(), [3, 1, 3]);
        assert_eq!(tensor, Tensor::from([[[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]]));
    }
}
