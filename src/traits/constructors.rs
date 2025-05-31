use crate::util::flatten::Flatten;
use crate::util::nested::Nested;
use crate::util::shape::Shape;
use crate::util::to_vec::ToVec;
use crate::{FloatDataType, NumericDataType, RawDataType, StridedMemory};
use num::NumCast;

pub trait Constructors<T: RawDataType>: StridedMemory {
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
    unsafe fn from_contiguous_owned_buffer(shape: Vec<usize>, data: Vec<T>) -> Self;

    /// Constructs an n-dimensional `NdArray` from input data such as a vector or array.
    ///
    /// # Parameters
    /// - `data`: a nested array or vector of valid data types (floats, integers, bools)
    ///
    /// # Panics
    ///   - If the input data has inhomogeneous dimensions, i.e., nested arrays do not have consistent sizes.
    ///   - If the input data is empty (cannot create a zero-length ndarray)
    ///
    /// # Example
    /// ```
    /// # use chela::*;
    ///
    /// let ndarray : NdArray<i32> = NdArray::new([[1, 2], [3, 4]]);
    /// assert_eq!(ndarray.shape(), &[2, 2]);
    ///
    /// let ndarray = NdArray::new(vec![1f32, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(ndarray.ndims(), 1);
    /// ```
    fn new<const D: usize>(data: impl Flatten<T> + Shape + Nested<{ D }>) -> Self {
        assert!(data.check_homogenous(), "from() failed, found inhomogeneous dimensions");

        let shape = data.shape();
        let data = data.flatten();

        assert!(!data.is_empty(), "from() failed, cannot create data buffer from empty data");

        unsafe { Self::from_contiguous_owned_buffer(shape, data) }
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
    fn full(n: T, shape: impl ToVec<usize>) -> Self {
        let shape = shape.to_vec();

        let data = vec![n; shape.iter().product()];
        assert!(!data.is_empty(), "cannot create an empty ndarray!");

        unsafe { Self::from_contiguous_owned_buffer(shape, data) }
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
    fn zeros(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>
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
    fn ones(shape: impl ToVec<usize>) -> Self
    where
        T: From<bool>
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
    fn scalar(n: T) -> Self {
        Self::full(n, [])
    }

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
    fn arange(start: T, stop: T) -> Self
    where
        T: NumericDataType
    {
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
    fn arange_with_step(start: T, stop: T, step: T) -> Self
    where
        T: NumericDataType
    {
        let n = ((stop - start).to_float() / step.to_float()).ceil();
        let n = NumCast::from(n).unwrap();

        let mut data: Vec<T> = vec![T::default(); n];
        for (i, item) in data.iter_mut().enumerate() {
            *item = <T as NumCast>::from(i).unwrap() * step + start;
        }

        unsafe { Self::from_contiguous_owned_buffer(vec![data.len()], data) }
    }

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
    /// assert_eq!(result, NdArray::new([0f32, 0.25, 0.5, 0.75, 1.0]));
    /// ```
    fn linspace(start: T, stop: T, num: usize) -> Self
    where
        T: FloatDataType
    {
        assert!(num > 0);

        if num == 1 {
            return unsafe { Self::from_contiguous_owned_buffer(vec![1], vec![start]) };
        }

        let step = (stop - start) / (<T as NumCast>::from(num).unwrap() - T::one());

        // from start to (stop + step) to make the range inclusive
        Self::arange_with_step(start, stop + step, step)
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
    /// assert_eq!(result, NdArray::new([0f32, 0.2, 0.4, 0.6, 0.8]));
    /// ```
    fn linspace_exclusive(start: T, stop: T, num: usize) -> Self
    where
        T: FloatDataType
    {
        assert!(num > 0);

        if num == 1 {
            return unsafe { Self::from_contiguous_owned_buffer(vec![1], vec![start]) };
        }

        let step = (stop - start) / <T as NumCast>::from(num).unwrap();
        Self::arange_with_step(start, stop, step)
    }
}
