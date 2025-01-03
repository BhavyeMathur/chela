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
        for ptr in self.flatiter_ptr() {
            unsafe { std::ptr::write(ptr, value); }
        }
    }

    // TODO we can probably make further optimisations in cases where the tensor isn't contiguous,
    // but where each element is located at a uniform stride from each other.
    // For example, let tensor = zeros(5, 2); view = tensor[::2
    // Then, each element of view is distributed with a stride of 2.
    // However, let tensor = zeros(4, 2); view = tensor[::2]
    // Then, view does not have a uniform stride because of elements at the boundary of axes 0 & 1

    pub fn fill(&mut self, value: T) {
        if self.is_contiguous() {
            return unsafe { self.fill_contiguous(value) };
        }
        self.fill_non_contiguous(value)
    }
}
