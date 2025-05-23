use crate::dtype::RawDataType;
use crate::iterator::util::split_by_indices;
use crate::tensor::flags::TensorFlags;
use crate::util::haslength::HasLength;
use crate::Tensor;

#[non_exhaustive]
pub struct NdIterator<'a, T: RawDataType> {
    result: Tensor<'a, T>,

    shape: Vec<usize>,
    stride: Vec<usize>,

    indices: Vec<usize>, // current index along each dimension
    iterator_index: usize,
    size: usize,
}

impl<T: RawDataType> Tensor<'_, T> {
    pub(crate) unsafe fn offset_ptr(&mut self, offset: isize) {
        self.ptr = self.ptr.offset(offset);
    }
}

impl<'a, T: RawDataType> NdIterator<'a, T> {
    pub(super) fn from<I>(tensor: &Tensor<'a, T>, axes: I) -> Self
    where
        I: IntoIterator<Item=usize> + HasLength + Clone,
    {
        let ndims = axes.len();
        let (shape, output_shape) = split_by_indices(&tensor.shape, axes.clone());
        let (stride, output_stride) = split_by_indices(&tensor.stride, axes);
        let size = shape.iter().product();

        Self {
            result: unsafe { tensor.reshaped_view(output_shape, output_stride) },
            shape,
            stride,
            indices: vec![0; ndims],
            iterator_index: 0,
            size,
        }
    }
}

impl<'a, T: RawDataType> Iterator for NdIterator<'a, T> {
    type Item = Tensor<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_index == self.size {
            return None;
        }

        let return_value = self.result.view();
        self.iterator_index += 1;
        
        for i in (0..self.shape.len()).rev() {
            if self.indices[i] != self.shape[i] {
                self.indices[i] += 1;
                unsafe { self.result.offset_ptr(self.stride[i] as isize); }
                break;
            }
        
            unsafe { self.result.offset_ptr(-((self.stride[i] * (self.shape[i] - 1)) as isize)); }
            self.indices[i] = 0;
        }

        Some(return_value)
    }
}
