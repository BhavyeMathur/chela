use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::iterator::util::split_by_indices;
use crate::tensor::flags::TensorFlags;
use crate::traits::haslength::HasLength;
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
    pub(in crate::tensor) unsafe fn offset_ptr(&mut self, offset: isize) {
        self.ptr = self.ptr.offset(offset);
    }

    pub fn has_uniform_stride(&self) -> Option<usize> {
        let (_, new_stride) = collapse_to_uniform_stride(&self.shape, &self.stride);

        // TODO don't need to calculate entire collapsed stride so this can be faster
        if new_stride.len() == 1 {
            return Some(new_stride[0]);
        }
        None
    }
}

impl<'a, T: RawDataType> NdIterator<'a, T> {
    pub(super) fn from<I>(tensor: &'a Tensor<T>, axes: I) -> Self
    where
        I: IntoIterator<Item=usize> + HasLength + Clone,
    {
        let ndims = axes.len();
        let (shape, output_shape) = split_by_indices(&tensor.shape, axes.clone());
        let (stride, output_stride) = split_by_indices(&tensor.stride, axes);
        let size = shape.iter().product();

        Self {
            result: unsafe { tensor.mut_reshaped_view(output_shape, output_stride) },
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

        let return_value = unsafe { self.result.lifetime_cast() };
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

impl<'a, T: RawDataType> Tensor<'a, T> {
    /// Creates a view of the tensor with arbitrary lifetime
    /// Safety: ensure returned tensor actually has a valid lifetime!
    unsafe fn lifetime_cast<'b>(&'a self) -> Tensor<'b, T> {
        Tensor {
            ptr: self.ptr,
            len: self.len,
            capacity: self.capacity,

            shape: self.shape.clone(),
            stride: self.stride.clone(),
            flags: self.flags - TensorFlags::Owned,

            _marker: Default::default(),
        }
    }
}
