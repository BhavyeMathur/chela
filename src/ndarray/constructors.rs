use crate::dtype::NumericDataType;
use crate::ndarray::flags::NdArrayFlags;
use crate::ndarray::NdArray;
use crate::util::flatten::Flatten;
use crate::util::nested::Nested;
use crate::util::shape::Shape;
use crate::util::to_vec::ToVec;
use crate::{FloatDataType, RawDataType, StridedMemory};
use num::NumCast;
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

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Constructs a new ndarray from the given data buffer and shape assuming a contiguous layout
    ///
    /// # Parameters
    /// - `shape`: A vector that defines the dimensions of the ndarray.
    /// - `data`: The underlying buffer that holds the ndarray's elements.
    /// - `requires_grad`: If gradients need to be computed for this ndarray.
    ///
    /// # Safety
    /// - `data` must remain valid and not be used elsewhere after being passed to this function.
    /// - `shape.iter().product()` must equal `data.len()`
    pub(crate) unsafe fn from_contiguous_owned_buffer(shape: Vec<usize>, data: Vec<T>) -> Self {
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

    /// Constructs an n-dimensional `NdArray` from input data such as a vector or array.
    ///
    /// # Parameters
    /// - `data`: a nested array or vector of valid data types (floats, integers, bools)
    ///
    /// # Panics
    ///   - If the input data has inhomogeneous dimensions, i.e., nested arrays do not have consistent sizes.
    ///   - If the input data is empty (cannot create a zero-length ndarray )
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray : NdArray<i32> = NdArray::from([[1, 2], [3, 4]]);
    /// assert_eq!(ndarray.shape(), &[2, 2]);
    ///
    /// let ndarray = NdArray::from(vec![1f32, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(ndarray.ndims(), 1);
    /// ```
    pub fn from<const D: usize>(data: impl Flatten<T> + Shape + Nested<{ D }>) -> Self {
        assert!(data.check_homogenous(), "Tensor::from() failed, found inhomogeneous dimensions");

        let shape = data.shape();
        let data = data.flatten();

        assert!(!data.is_empty(), "Tensor::from() failed, cannot create data buffer from empty data");

        unsafe { NdArray::from_contiguous_owned_buffer(shape, data) }
    }

    /// Creates an ndarray filled with a specified value and given shape.
    ///
    /// # Parameters
    ///
    /// * `n` - The value to fill the ndarray with (can be any valid data type like float, integer, or bool).
    /// * `shape` - An array or vector representing the shape of the ndarray (e.g. `[2, 3, 5]`).
    ///
    /// # Panics
    /// This function panics if the provided shape is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::full(5i32, [2, 3]); // creates a 2x3 ndarray filled with the value 5.
    /// let ndarray = NdArray::full(true, [2, 3, 5]); // creates a 2x3x5 ndarray filled with 'true'
    /// ```
    pub fn full(n: T, shape: impl ToVec<usize>) -> Self {
        let shape = shape.to_vec();

        let data = vec![n; shape.iter().product()];
        assert!(!data.is_empty(), "Cannot create an empty tensor!");

        unsafe { NdArray::from_contiguous_owned_buffer(shape, data) }
    }

    /// Creates a new ndarray filled with zeros with the given shape.
    ///
    /// # Parameters
    /// - `shape`: An array or vector representing the shape of the ndarray (e.g. `[2, 3, 5]`).
    ///
    /// # Panics
    /// This function panics if the provided shape is empty.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::<i32>::zeros([2, 3]);
    /// let ndarray = NdArray::<bool>::zeros([2, 3]);  // creates an ndarray filled with 'false'
    /// ```
    pub fn zeros(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>,
    {
        Self::full(false.into(), shape)
    }

    /// Creates a new ndarray filled with ones with the given shape.
    ///
    /// # Parameters
    /// - `shape`: An array or vector representing the shape of the ndarray (e.g. `[2, 3, 5]`).
    ///
    /// # Panics
    /// This function panics if the provided shape is empty.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::<i32>::ones([2, 3]);
    /// let ndarray = NdArray::<bool>::ones([2, 3]);  // creates an ndarray filled with 'true'
    /// ```
    pub fn ones(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>,
    {
        Self::full(true.into(), shape)
    }

    /// Creates a 0-dimensional (shapeless) ndarray containing a single value.
    ///
    /// # Parameters
    /// - `n`: The value to be stored in the scalar ndarray.
    ///
    /// # Example
    /// ```rust
    /// # use chela::*;
    ///
    /// let scalar_array = NdArray::scalar(42);
    /// assert_eq!(scalar_array.shape(), []);
    /// assert_eq!(scalar_array.value(), 42);
    /// ```
    pub fn scalar(n: T) -> Self {
        NdArray::full(n, [])
    }

    // Maybe we should support empty arrays one day.
    // pub fn empty() -> Self {
    //     unsafe { NdArray::from_contiguous_owned_buffer(vec![0], vec![]) }
    // }

    /// Retrieves the single value contained within an ndarray with a singular element.
    ///
    /// # Panics
    /// If the ndarray contains more than one element (i.e., it is not a scalar or an ndarray with a
    /// single element)
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray = NdArray::scalar(50f32);
    /// let value = ndarray.value();
    /// assert_eq!(value, 50.0);
    /// ```
    ///
    /// # Notes
    /// This function is only meant for arrays that are guaranteed to have
    /// exactly one element. For arrays with multiple elements, consider using
    /// appropriate methods to access individual elements or slices safely.
    pub fn value(&self) -> T {
        assert_eq!(self.size(), 1, "cannot get value of a tensor with more than one element");
        unsafe { self.ptr.read() }
    }
}

impl<T: NumericDataType> NdArray<'_, T> {
    /// Generates a 1D ndarray with evenly spaced values within a specified range.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the sequence, inclusive.
    /// * `stop` - The ending value of the sequence, exclusive.
    ///
    /// # Returns
    ///
    /// An `NdArray` containing values starting from `start` and ending before `stop`,
    /// with a step-size of 1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::arange(0i32, 5); // [0, 1, 2, 3, 4].
    /// ```
    pub fn arange(start: T, stop: T) -> NdArray<'static, T> {
        Self::arange_with_step(start, stop, T::one())
    }

    /// Generates a 1D ndarray with evenly spaced values within a specified range.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the sequence, inclusive.
    /// * `stop` - The ending value of the sequence, exclusive.
    /// * `step` - The interval between each consecutive value
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    /// let ndarray = NdArray::arange_with_step(0i32, 5, 2); // [0, 2, 4].
    /// ```
    pub fn arange_with_step(start: T, stop: T, step: T) -> NdArray<'static, T> {
        let n = ((stop - start).to_float() / step.to_float()).ceil();
        let n = NumCast::from(n).unwrap();

        let mut data: Vec<T> = vec![T::default(); n];
        for (i, item) in data.iter_mut().enumerate() {
            *item = <T as NumCast>::from(i).unwrap() * step + start;
        }

        unsafe { NdArray::from_contiguous_owned_buffer(vec![data.len()], data) }
    }
}

impl<T: FloatDataType> NdArray<'_, T> {
    /// Generates a 1-dimensional ndarray with `num `evenly spaced values between `start` and `stop`
    /// (inclusive).
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the sequence.
    /// * `stop` - The ending value of the sequence. The value is inclusive in the range.
    /// * `num` - The number of evenly spaced values to generate. Must be greater than 0.
    ///
    /// # Panic
    ///
    /// Panics if `num` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// # use chela::*;
    /// let result = NdArray::linspace(0f32, 1.0, 5);  // [0.0, 0.25, 0.5, 0.75, 1.0]
    /// assert_eq!(result, NdArray::from([0f32, 0.25, 0.5, 0.75, 1.0]));
    /// ```
    pub fn linspace(start: T, stop: T, num: usize) -> NdArray<'static, T> {
        assert!(num > 0);

        if num == 1 {
            return unsafe { NdArray::from_contiguous_owned_buffer(vec![1], vec![start]) };
        }

        let step = (stop - start) / (<T as NumCast>::from(num).unwrap() - T::one());

        // from start to (stop + step) to make the range inclusive
        NdArray::arange_with_step(start, stop + step, step)
    }

    /// Generates a 1-dimensional ndarray with `num `evenly spaced values between `start` and `stop`
    /// (exclusive).
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the sequence.
    /// * `stop` - The ending value of the sequence. The value is exclusive in the range.
    /// * `num` - The number of evenly spaced values to generate. Must be greater than 0.
    ///
    /// # Panic
    ///
    /// Panics if `num` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// # use chela::*;
    /// let result = NdArray::linspace_exclusive(0.0f32, 1.0, 5);
    /// assert_eq!(result, NdArray::from([0f32, 0.2, 0.4, 0.6, 0.8]));
    /// ```
    pub fn linspace_exclusive(start: T, stop: T, num: usize) -> NdArray<'static, T> {
        assert!(num > 0);

        if num == 1 {
            return unsafe { NdArray::from_contiguous_owned_buffer(vec![1], vec![start]) };
        }

        let step = (stop - start) / <T as NumCast>::from(num).unwrap();
        NdArray::arange_with_step(start, stop, step)
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
