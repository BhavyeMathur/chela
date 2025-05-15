use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::tensor::flags::TensorFlags;
use crate::Tensor;

pub trait TensorMethods {
    fn shape(&self) -> &[usize];

    fn stride(&self) -> &[usize];

    fn ndims(&self) -> usize {
        self.shape().len()
    }

    #[inline]
    fn len(&self) -> usize {
        if self.shape().is_empty() {
            return 0;
        }

        self.shape()[0]
    }

    #[inline]
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn flags(&self) -> TensorFlags;

    #[inline]
    fn is_contiguous(&self) -> bool {
        self.flags().contains(TensorFlags::Contiguous)
    }

    #[inline]
    fn is_view(&self) -> bool {
        !self.flags().contains(TensorFlags::Owned)
    }

    #[inline]
    fn has_uniform_stride(&self) -> Option<usize> {
        if !self.flags().contains(TensorFlags::UniformStride) {
            return None;
        }

        if self.ndims() == 0 {
            return Some(0);
        }

        let (_, new_stride) = collapse_to_uniform_stride(self.shape(), self.stride());
        Some(new_stride[0])
    }
}

impl<T: RawDataType> TensorMethods for Tensor<'_, T> {
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn stride(&self) -> &[usize] {
        &self.stride
    }

    #[inline]
    fn flags(&self) -> TensorFlags {
        self.flags
    }
}
