use crate::dtype::NumericDataType;
use crate::tensor::dtype::RawDataType;
use crate::tensor::flags::TensorFlags;
use crate::tensor::Tensor;
use crate::traits::flatten::Flatten;
use crate::traits::nested::Nested;
use crate::traits::shape::Shape;
use crate::traits::to_vec::ToVec;
use crate::{FloatDataType, TensorMethods};
use num::NumCast;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

/// Computes the stride of a tensor from its given shape assuming a contiguous layout.
///
/// In the context of multidimensional tensors, the stride refers to the number of elements
/// that need to be skipped in memory to move to the next element along each dimension.
/// Strides are calculated by determining how many elements are spanned by the dimensions
/// following a particular axis.
///
/// # Arguments
///
/// * `shape` - A slice representing the shape of the tensor.
///
/// # Returns
///
/// A `Vec<usize>` containing the stride for each dimension of the tensor, with the same
/// length as the input `shape`. The result indicates how many elements need to be skipped
/// in memory to traverse the tensor along each dimension.
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

impl<T: RawDataType> Tensor<'_, T> {
    /// Constructs a new tensor from the given data buffer and metadata (shape and stride).
    ///
    /// # Parameters
    /// - `shape`: A vector that defines the dimensions of the tensor.
    /// - `stride`: A vector that defines the strides of the tensor.
    /// - `data`: The underlying buffer that holds the tensor's elements.
    ///
    /// # Safety
    /// - `data` must remain valid and not be used elsewhere after being passed to this function.
    /// - `stride` and `shape` must describe a valid (not necessarily contiguous) memory layout for `data`
    pub(super) unsafe fn from_owned_buffer(
        shape: Vec<usize>,
        stride: Vec<usize>,
        data: Vec<T>,
    ) -> Self {
        // take control of the data so that Rust doesn't drop it once the vector goes out of scope
        let mut data = ManuallyDrop::new(data);

        Self {
            ptr: NonNull::new_unchecked(data.as_mut_ptr()),
            len: data.len(),
            capacity: data.capacity(),

            shape,
            stride,
            flags: TensorFlags::Owned | TensorFlags::Contiguous | TensorFlags::UniformStride | TensorFlags::Writeable,

            _marker: Default::default(),
        }
    }

    /// Constructs a new tensor from the given data buffer and shape assuming a contiguous layout
    ///
    /// # Parameters
    /// - `shape`: A vector that defines the dimensions of the tensor.
    /// - `data`: The underlying buffer that holds the tensor's elements.
    ///
    /// # Safety
    /// - `data` must remain valid and not be used elsewhere after being passed to this function.
    /// - `shape.iter().product()` must equal `data.len()`
    pub(crate) unsafe fn from_contiguous_owned_buffer(shape: Vec<usize>, data: Vec<T>) -> Self {
        let stride = stride_from_shape(&shape);
        Self::from_owned_buffer(shape, stride, data)
    }

    /// Constructs an n-dimensional `Tensor` from input data such as a vector or array.
    ///
    /// # Parameters
    /// - `data`: a nested array or vector of valid data types (floats, integers, bools)
    ///
    /// # Panics
    ///   - If the input data has inhomogeneous dimensions, i.e., nested arrays do not have consistent sizes.
    ///   - If the input data is empty (cannot create a zero-length tensor)
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let tensor: Tensor<i32> = Tensor::from([[1, 2], [3, 4]]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    ///
    /// let tensor = Tensor::from(vec![1f32, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(tensor.ndims(), 1);
    /// ```
    pub fn from<const D: usize>(data: impl Flatten<T> + Shape + Nested<{ D }>) -> Self {
        assert!(
            data.check_homogenous(),
            "Tensor::from() failed, found inhomogeneous dimensions"
        );

        let shape = data.shape();
        let data = data.flatten();

        assert!(
            !data.is_empty(),
            "Tensor::from() failed, cannot create data buffer from empty data"
        );

        unsafe { Tensor::from_contiguous_owned_buffer(shape, data) }
    }

    /// Creates a tensor filled with a specified value and given shape.
    ///
    /// # Parameters
    ///
    /// * `n` - The value to fill the tensor with (can be any valid data type like float, integer, or bool).
    /// * `shape` - An array or vector representing the shape of the tensor (e.g. `[2, 3, 5]`).
    ///
    /// # Panics
    /// This function panics if the provided shape is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = Tensor::full(5i32, [2, 3]); // creates a 2x3 tensor filled with the value 5.
    /// let tensor = Tensor::full(true, [2, 3, 5]); // creates a 2x3x5 tensor filled with 'true'
    /// ```
    pub fn full(n: T, shape: impl ToVec<usize>) -> Self {
        let shape = shape.to_vec();

        let data = vec![n; shape.iter().product()];
        assert!(!data.is_empty(), "Cannot create an empty tensor!");

        unsafe { Tensor::from_contiguous_owned_buffer(shape, data) }
    }

    /// Creates a new tensor filled with zeros with the given shape.
    ///
    /// # Parameters
    /// - `shape`: An array or vector representing the shape of the tensor (e.g. `[2, 3, 5]`).
    ///
    /// # Panics
    /// This function panics if the provided shape is empty.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = Tensor::<i32>::zeros([2, 3]);
    /// let tensor = Tensor::<bool>::zeros([2, 3]);  // creates a tensor filled with 'false'
    /// ```
    pub fn zeros(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>,
    {
        Self::full(false.into(), shape)
    }

    /// Creates a new tensor filled with ones with the given shape.
    ///
    /// # Parameters
    /// - `shape`: An array or vector representing the shape of the tensor (e.g. `[2, 3, 5]`).
    ///
    /// # Panics
    /// This function panics if the provided shape is empty.
    ///
    /// # Examples
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = Tensor::<i32>::ones([2, 3]);
    /// let tensor = Tensor::<bool>::ones([2, 3]);  // creates a tensor filled with 'true'
    /// ```
    pub fn ones(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>,
    {
        Self::full(true.into(), shape)
    }


    /// Creates a 0-dimensional (shapeless) tensor containing a single value.
    ///
    /// # Parameters
    /// - `n`: The value to be stored in the scalar tensor.
    ///
    /// # Example
    /// ```rust
    /// # use chela::*;
    ///
    /// let scalar_tensor = Tensor::scalar(42);
    /// assert_eq!(scalar_tensor.shape(), []);
    /// assert_eq!(scalar_tensor.value(), 42);
    /// ```
    pub fn scalar(n: T) -> Self {
        Tensor::full(n, [])
    }

    /// Retrieves the single value contained within a tensor with a singular element.
    ///
    /// # Panics
    /// If the tensor contains more than one element (i.e., it is not a scalar or a tensor with a
    /// single element)
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let tensor = Tensor::scalar(50f32);
    /// let value = tensor.value();
    /// assert_eq!(value, 50.0);
    /// ```
    ///
    /// # Notes
    /// This function is only meant for tensors that are guaranteed to have
    /// exactly one element. For tensors with multiple elements, consider using
    /// appropriate methods to access individual elements or slices safely.
    pub fn value(&self) -> T {
        assert_eq!(self.size(), 1, "cannot get value of a tensor with more than one element");
        unsafe { self.ptr.read() }
    }
}

impl<T: NumericDataType> Tensor<'_, T> {
    /// Generates a 1D tensor with evenly spaced values within a specified range.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the sequence, inclusive.
    /// * `stop` - The ending value of the sequence, exclusive.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing values starting from `start` and ending before `stop`,
    /// with a step-size of 1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    /// let tensor = Tensor::arange(0i32, 5); // [0, 1, 2, 3, 4].
    /// ```
    pub fn arange(start: T, stop: T) -> Tensor<'static, T> {
        let n = NumCast::from((stop - start).ceil()).unwrap();

        let mut data: Vec<T> = vec![T::default(); n];
        for i in 0..n {
            data[i] = <T as NumCast>::from(i).unwrap() + start;
        }

        unsafe { Tensor::from_contiguous_owned_buffer(vec![data.len()], data) }
    }

    /// Generates a 1D tensor with evenly spaced values within a specified range.
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
    /// let tensor = Tensor::arange_with_step(0i32, 5, 2); // [0, 2, 4].
    /// ```
    pub fn arange_with_step(start: T, stop: T, step: T) -> Tensor<'static, T> {
        let n = ((stop - start).to_float() / step.to_float()).ceil();
        let n = NumCast::from(n).unwrap();

        let mut data: Vec<T> = vec![T::default(); n];
        for i in 0..n {
            data[i] = <T as NumCast>::from(i).unwrap() * step + start;
        }

        unsafe { Tensor::from_contiguous_owned_buffer(vec![data.len()], data) }
    }
}

impl<T: FloatDataType> Tensor<'_, T> {
    /// Generates a 1-dimensional tensor with `num `evenly spaced values between `start` and `stop`
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
    /// let result = Tensor::linspace(0f32, 1.0, 5);  // [0.0, 0.25, 0.5, 0.75, 1.0]
    /// assert_eq!(result, Tensor::from([0f32, 0.25, 0.5, 0.75, 1.0]));
    /// ```
    pub fn linspace(start: T, stop: T, num: usize) -> Tensor<'static, T> {
        assert!(num > 0);

        if num == 1 {
            return unsafe { Tensor::from_contiguous_owned_buffer(vec![1], vec![start]) };
        }

        let step = (stop - start) / (T::from(num).unwrap() - T::one());

        // from start to (stop + step) to make the range inclusive
        Tensor::arange_with_step(start, stop + step, step)
    }

    /// Generates a 1-dimensional tensor with `num `evenly spaced values between `start` and `stop`
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
    /// let result = Tensor::linspace_exclusive(0.0f32, 1.0, 5);
    /// assert_eq!(result, Tensor::from([0f32, 0.2, 0.4, 0.6, 0.8]));
    /// ```
    pub fn linspace_exclusive(start: T, stop: T, num: usize) -> Tensor<'static, T> {
        assert!(num > 0);

        if num == 1 {
            return unsafe { Tensor::from_contiguous_owned_buffer(vec![1], vec![start]) };
        }

        let step = (stop - start) / T::from(num).unwrap();
        Tensor::arange_with_step(start, stop, step)
    }
}

impl<T: RawDataType> Drop for Tensor<'_, T> {
    /// This method is implicitly invoked when the tensor is deleted to clean up its memory if
    /// the tensor owns its data (i.e. it is not a view into another tensor).
    /// 
    /// Resets `self.len` and `self.capacity` to 0.
    fn drop(&mut self) {
        if self.flags.contains(TensorFlags::Owned) {
            // drops the data
            unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };
        }

        self.len = 0;
        self.capacity = 0;
    }
}
