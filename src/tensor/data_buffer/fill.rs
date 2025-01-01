use crate::accelerate::cblas::{catlas_dset, catlas_sset};
use crate::data_buffer::{DataBuffer, DataOwned};
use crate::dtype::RawDataType;
use std::ffi::c_int;

pub(in crate::tensor) trait Fill<T>
where
    T: RawDataType,
{
    fn fill(&mut self, value: T);

    fn fill_naive(&mut self, value: T);
}

#[cfg(target_vendor = "apple")]
impl Fill<f32> for DataOwned<f32> {
    fn fill(&mut self, value: f32) {
        unsafe { catlas_sset(self.len as c_int, value, self.const_ptr(), 1) }
    }

    #[inline(never)]
    fn fill_naive(&mut self, value: f32) {
        let mut ptr = self.mut_ptr();
        let end_ptr = unsafe { ptr.add(self.len) };

        while ptr != end_ptr {
            unsafe {
                std::ptr::write(ptr, value);
                ptr = ptr.add(1);
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
impl Fill<f64> for DataOwned<f64> {
    fn fill(&mut self, value: f64) {
        unsafe { catlas_dset(self.len as c_int, value, self.const_ptr(), 1) }
    }

    fn fill_naive(&mut self, value: f64) {
        let mut ptr = self.mut_ptr();
        let end_ptr = unsafe { ptr.add(self.len) };

        while ptr != end_ptr {
            unsafe {
                std::ptr::write(ptr, value);
                ptr = ptr.add(1);
            }
        }
    }
}
