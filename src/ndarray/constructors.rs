use crate::ndarray::flags::NdArrayFlags;
use crate::ndarray::NdArray;
use crate::traits::constructors::Constructors;
use crate::RawDataType;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

/// Computes the stride of an ndarray from its given shape assuming a contiguous layout.
///
/// In the context of multidimensional arrays, the stride refers to the number of elements
/// that need to be skipped in memory to move to the next element along each dimension.
/// Strides are calculated by determining how many elements are spanned by the dimensions
/// following a particular axis.
///
/// # Arguments
///
/// * `shape` - A slice representing the shape of the ndarray.
///
/// # Returns
///
/// A `Vec<usize>` containing the stride for each dimension of the ndarray, with the same
/// length as the input `shape`. The result indicates how many elements need to be skipped
/// in memory to traverse the ndarray along each dimension.
///
/// # Example
///
/// ```
/// let shape = vec![5, 3, 2, 1];
///
/// // stride would be [10, 2, 1, 1]
/// // Axis 0 (size 5): stride = 3 * 2 * 1 * 1 = 10
/// // Axis 1 (size 3): stride = 2 * 1 * 1 = 2
/// // Axis 2 (size 2): stride = 1 * 1
/// // Axis 3 (size 1): stride is always 1
/// ```
pub(crate) fn stride_from_shape(shape: &[usize]) -> Vec<usize> {
    let ndims = shape.len();
    let mut stride = vec![0; ndims];

    let mut p = 1;
    for i in (0..ndims).rev() {
        stride[i] = p;
        p *= shape[i];
    }

    stride
}


impl<'a, T: RawDataType> Constructors<T> for NdArray<'a, T> {
    unsafe fn from_contiguous_owned_buffer(shape: Vec<usize>, data: Vec<T>) -> Self {
        let flags = NdArrayFlags::Owned | NdArrayFlags::Contiguous | NdArrayFlags::UniformStride | NdArrayFlags::Writeable;

        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
        let mut data = ManuallyDrop::new(data);
        let stride = stride_from_shape(&shape);

        Self {
            ptr: NonNull::new_unchecked(data.as_mut_ptr()),
            len: data.len(),
            capacity: data.capacity(),

            shape,
            stride,
            flags,

            _marker: Default::default(),
        }
    }
}

impl<T: RawDataType> Drop for NdArray<'_, T> {
    /// This method is implicitly invoked when the ndarray is deleted to clean up its memory if
    /// the ndarray owns its data (i.e. it is not a view into another ndarray ).
    ///
    /// Resets `self.len` and `self.capacity` to 0.
    fn drop(&mut self) {
        if self.flags.contains(NdArrayFlags::Owned) {
            // drops the data
            unsafe { Vec::from_raw_parts(self.mut_ptr(), self.len, self.capacity) };
        }

        self.len = 0;
        self.capacity = 0;
    }
}
