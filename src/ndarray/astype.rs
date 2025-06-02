use crate::dtype::NumericDataType;
use crate::{Constructors, NdArray, StridedMemory};
use num::NumCast;
use crate::ndarray::flags::NdArrayFlags;

impl<T: NumericDataType> NdArray<'_, T> {
    pub fn astype<'b, F: NumericDataType>(&self) -> NdArray<'b, F>
    {
        let mut data = vec![F::default(); self.size()];

        for (dst, src) in data.iter_mut().zip(self.flatiter()) {
            *dst = NumCast::from(src).expect("astype conversion failed");
        }
        
        unsafe { NdArray::from_contiguous_owned_buffer(self.shape().to_vec(), data) }
    }

    pub(crate) unsafe fn lifetime_cast(&self) -> NdArray<'static, T> {
        NdArray {
            ptr: self.ptr,
            len: self.len,
            capacity: 0,

            shape: self.shape.clone(),
            stride: self.stride.clone(),
            flags: self.flags - NdArrayFlags::UserCreated,
            
            _marker: Default::default(),
        }
    }
}
