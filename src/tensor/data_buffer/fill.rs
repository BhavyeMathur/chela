use crate::data_buffer::{DataBuffer, DataOwned};
use crate::dtype::RawDataType;

pub(in crate::tensor) trait Fill<T> {
    fn fill(&mut self, value: T);
}

impl<T> Fill<T> for DataOwned<T>
where
    T: RawDataType,
{
    fn fill(&mut self, value: T) {
        unsafe {
            let mut ptr = self.mut_ptr();
            let end_ptr = ptr.add(self.len);

            while ptr != end_ptr {
                std::ptr::write(ptr, value);
                ptr = ptr.add(1);
            }
        }
    }
}
