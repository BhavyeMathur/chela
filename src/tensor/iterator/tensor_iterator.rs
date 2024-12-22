use crate::data_buffer::DataBuffer;
use crate::dtype::RawDataType;
use crate::iterator::util::split_by_indices;
use crate::traits::haslength::HasLength;
use crate::{Axis, Tensor, TensorBase, TensorView};

#[non_exhaustive]
pub struct TensorIterator<T>
where
    T: RawDataType,
{
    result: TensorView<T>,

    shape: Vec<usize>,
    stride: Vec<usize>,
    ndims: usize,

    indices: Vec<usize>, // current index along each dimension
    iterator_index: usize,
    size: usize,
}

impl<T: RawDataType> Tensor<T> {
    pub fn iter(&self) -> TensorIterator<T> {
        TensorIterator::from(self, [0])
    }

    pub fn iter_along(&self, axis: Axis) -> TensorIterator<T> {
        TensorIterator::from(self, [axis.0])
    }

    pub fn nditer(&self, axes: impl IntoIterator<Item=usize> + HasLength + Clone) -> TensorIterator<T> {
        TensorIterator::from(self, axes)
    }
}

impl<T> TensorIterator<T>
where
    T: RawDataType,
{
    pub(super) fn from<B, I>(tensor: &TensorBase<B>, axes: I) -> Self
    where
        B: DataBuffer<DType=T>,
        I: IntoIterator<Item=usize> + HasLength + Clone,
    {
        let ndims = axes.len();
        let (shape, output_shape) = split_by_indices(&tensor.shape, axes.clone());
        let (stride, output_stride) = split_by_indices(&tensor.stride, axes);
        let size = shape.iter().product();

        Self {
            result: TensorView::from(tensor, 0, output_shape, output_stride),
            shape,
            stride,
            ndims,
            indices: vec![0; ndims],
            iterator_index: 0,
            size,
        }
    }
}

impl<T> Iterator for TensorIterator<T>
where
    T: RawDataType,
{
    type Item = TensorView<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_index == self.size {
            return None;
        }

        let return_value = self.result.copy_view();
        self.iterator_index += 1;

        for i in (0..self.ndims).rev() {
            if self.indices[i] != self.shape[i] {
                self.indices[i] += 1;
                unsafe { self.result.data.add_offset(self.stride[i] as isize); }
                break;
            }

            unsafe { self.result.data.add_offset(-((self.stride[i] * (self.shape[i] - 1)) as isize)); }
            self.indices[i] = 0;
        }

        Some(return_value)
    }
}
