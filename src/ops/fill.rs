use crate::ndarray::collapse_contiguous::has_uniform_stride;

pub(crate) trait Fill: Copy {
    /// Fills a contiguous sequence of memory locations with a given value.
    ///
    /// # Safety
    /// - Ensure `ptr` describes a valid array with length `count`.
    ///
    /// # Parameters
    /// * `ptr`: A mutable raw pointer to the starting location of memory to be filled.
    /// * `count`: The number of values to write.
    /// * `value`: The value to write to the memory locations.
    unsafe fn fill_contiguous(ptr: *mut Self, count: usize, value: Self) {
        Self::fill_uniform_stride(ptr, count, 1, value);
    }

    /// Fills a sequence of memory locations with a given value, following a specified stride.
    ///
    /// # Safety
    /// - Ensure the memory described by `ptr`, `stride`, and `count` is valid.
    ///
    /// # Parameters
    /// * `ptr`: A mutable raw pointer to the starting location of memory to be filled.
    /// * `count`: The number of values to write.
    /// * `stride`: The stride (in number of elements of type `T`) between consecutive writes.
    /// * `value`: The value to write to the memory locations.
    unsafe fn fill_uniform_stride(mut ptr: *mut Self, count: usize, stride: usize, value: Self) {
        for _ in 0..count {
            *ptr = value;
            ptr = ptr.add(stride);
        }
    }

    /// Fills memory described by the layout of `shape` and `stride` with a given value.
    ///
    /// # Safety
    /// - Ensure the memory described by `ptr`, `shape`, and `stride` is valid.
    ///
    /// # Parameters
    /// * `ptr`: A mutable raw pointer to the starting location of memory to be filled.
    /// * `shape`: The shape along each dimension of the memory layout
    /// * `stride`: The stride between elements along each dimension of the memory layout
    /// * `len`: The total length of the memory layout.
    /// * `value`: The value to write to the memory locations.
    unsafe fn fill(mut ptr: *mut Self, shape: &[usize], stride: &[usize], len: usize, value: Self) {
        if let Some(stride) = has_uniform_stride(shape, stride) {
            return if stride == 1 {
                Self::fill_contiguous(ptr, len, value)
            } else {
                Self::fill_uniform_stride(ptr, shape.iter().product(), stride, value)
            }
        }

        for _ in 0..shape[0] {
            let len = shape[1] * stride[1];

            Self::fill(ptr, &shape[1..], &stride[1..], len, value);
            ptr = ptr.add(stride[0]);
        }
    }
}

impl Fill for bool {}

impl Fill for i8 {}
impl Fill for i16 {}

impl Fill for i64 {}
impl Fill for i128 {}
impl Fill for isize {}

impl Fill for u8 {}
impl Fill for u16 {}
impl Fill for u64 {}
impl Fill for u128 {}
impl Fill for usize {}


impl Fill for u32 {
    #[cfg(apple_vdsp)]
    unsafe fn fill_uniform_stride(ptr: *mut Self, count: usize, stride: usize, value: Self) {
        use crate::acceleration::vdsp::vDSP_vfilli;
        use std::ptr::addr_of;

        let value_i32 = unsafe { std::mem::transmute::<u32, i32>(value) };
        vDSP_vfilli(addr_of!(value_i32), ptr as *mut i32, stride as isize, count);
    }
}


impl Fill for i32 {
    #[cfg(apple_vdsp)]
    unsafe fn fill_uniform_stride(ptr: *mut Self, count: usize, stride: usize, value: Self) {
        use crate::acceleration::vdsp::vDSP_vfilli;
        use std::ptr::addr_of;
        
        vDSP_vfilli(addr_of!(value), ptr, stride as isize, count);
    }
}

impl Fill for f32 {
    #[cfg(apple_vdsp)]
    unsafe fn fill_uniform_stride(ptr: *mut Self, count: usize, stride: usize, value: Self) {
        use crate::acceleration::vdsp::vDSP_vfill;
        use std::ptr::addr_of;
        
        vDSP_vfill(addr_of!(value), ptr, stride as isize, count);
    }
}

impl Fill for f64 {
    #[cfg(apple_vdsp)]
    unsafe fn fill_uniform_stride(ptr: *mut Self, count: usize, stride: usize, value: Self) {
        use crate::acceleration::vdsp::vDSP_vfillD;
        use std::ptr::addr_of;
        
        vDSP_vfillD(addr_of!(value), ptr, stride as isize, count);
    }
}
