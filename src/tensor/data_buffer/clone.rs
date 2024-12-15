use std::intrinsics::copy_nonoverlapping;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use crate::data_buffer::DataOwned;
use crate::dtype::RawDataType;

impl<T: RawDataType> Clone for DataOwned<T> {
    fn clone(&self) -> DataOwned<T> {
        let mut data = Vec::with_capacity(self.len);

        let src = self.ptr.as_ptr();
        let dst = data.as_mut_ptr();

        unsafe {
            copy_nonoverlapping(src, dst, self.len);
            data.set_len(self.len);
        }

        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
        let data = ManuallyDrop::new(data);

        Self {
            len: data.len(),
            capacity: data.capacity(),
            ptr: NonNull::new(dst).unwrap(),
        }
    }
}
