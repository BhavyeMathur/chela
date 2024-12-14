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

impl<T: RawDataType> TensorView<T> {
    fn flat_indices(&self) -> Vec<usize> {
        let (shape, stride) = collapse_contiguous(&self.shape, &self.stride);
        let ndims = shape.len();

        let size = self.size();
        let mut flat_indices = Vec::with_capacity(size);
        let mut indices = vec![0; ndims];
        let mut flat_index = 0;

        for _ in 0..size {
            flat_indices.push(flat_index);

            for i in (0..ndims).rev() {
                indices[i] += 1;

                if indices[i] < shape[i] {
                    flat_index += stride[i];
                    break;
                }

                flat_index -= stride[i] * (shape[i] - 1);
                indices[i] = 0; // reset this dimension and carry over to the next
            }
        }

        flat_indices
    }
}

#[cfg(test)]
mod tests {
    use crate::flatten::collapse_contiguous;
    use crate::{s, Tensor};

    #[test]
    fn flat_indices() {
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

        let indices = b.flat_indices();
        assert_eq!(indices, [0, 1, 2, 6, 7, 8, 12, 13, 14]);

        let b = a.slice(s![1]);
        let (shape, stride) = collapse_contiguous(&b.shape, &b.stride);
        assert_eq!(shape, [6]);
        assert_eq!(stride, [1]);

        let indices = b.flat_indices();
        assert_eq!(indices, [0, 1, 2, 3, 4, 5]);

        let b = a.slice(s![..2, 1, 1..]);
        let (shape, stride) = collapse_contiguous(&b.shape, &b.stride);
        assert_eq!(shape, [2, 2]);
        assert_eq!(stride, [6, 1]);

        let indices = b.flat_indices();
        assert_eq!(indices, [0, 1, 6, 7]);
    }
}
