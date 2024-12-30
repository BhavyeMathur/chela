use crate::data_buffer::{DataBuffer, DataOwned};
use crate::dtype::RawDataType;
use std::ffi::c_int;

#[cfg(target_vendor = "apple")]
use crate::accelerate::cblas::{catlas_dset, catlas_sset};

pub(in crate::tensor) trait Fill<T>
where
    T: RawDataType,
{
    fn fill(&self, value: T);
}

#[cfg(target_vendor = "apple")]
impl Fill<f32> for DataOwned<f32> {
    fn fill(&self, value: f32) {
        unsafe { catlas_sset(self.len as c_int, value, self.const_ptr(), 1) }
    }
}

#[cfg(target_vendor = "apple")]
impl Fill<f64> for DataOwned<f64> {
    fn fill(&self, value: f64) {
        unsafe { catlas_dset(self.len as c_int, value, self.const_ptr(), 1) }
    }
}
