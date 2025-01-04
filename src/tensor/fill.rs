use crate::dtype::RawDataType;
use crate::Tensor;

impl<T: RawDataType> Tensor<T> {
    /// Safety: expects tensor buffer is contiguously stored
    unsafe fn fill_contiguous(&mut self, value: T) {
        let mut ptr = self.ptr.as_ptr();
        let end_ptr = ptr.add(self.len);

        while ptr != end_ptr {
            std::ptr::write(ptr, value);
            ptr = ptr.add(1);
        }
    }

    fn fill_non_contiguous(&mut self, value: T) {
        todo!()
    }

    pub fn fill(&mut self, value: T) {
        if self.is_contiguous() {
            return unsafe { self.fill_contiguous(value) };
        }
        self.fill_non_contiguous(value)
    }
}
