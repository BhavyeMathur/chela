use crate::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;
use crate::Tensor;

impl<T: RawDataType> Tensor<T> {
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    #[inline]
    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }

        self.shape[0]
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.flags.contains(TensorFlags::Contiguous)
    }

    #[inline]
    pub fn is_view(&self) -> bool {
        !self.flags.contains(TensorFlags::Owned)
    }
}
