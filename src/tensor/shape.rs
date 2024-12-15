use crate::data_buffer::DataBuffer;
use crate::TensorBase;

impl<T: DataBuffer> TensorBase<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn ndims(&self) -> usize {
        self.ndims
    }

    pub fn len(&self) -> &usize {
        if self.shape.len() == 0 {
            return &0;
        }

        &self.shape[0]
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}
