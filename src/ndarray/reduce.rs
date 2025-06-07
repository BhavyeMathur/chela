use crate::dtype::{NumericDataType, RawDataType};
use crate::flat_index_generator::FlatIndexGenerator;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::ops::reduce_max::ReduceMax;
use crate::ops::reduce_min::ReduceMin;
use crate::ops::reduce_min_magnitude::ReduceMinMagnitude;
use crate::ops::reduce_product::ReduceProduct;
use crate::ops::reduce_sum::ReduceSum;
use crate::partial_ord::*;
use crate::util::to_vec::ToVec;
use crate::{AxisType, Constructors, FloatDataType, NdArray, StridedMemory};
use num::NumCast;
use std::collections::VecDeque;
use crate::ops::reduce_max_magnitude::ReduceMaxMagnitude;

/// Returns a tuple `(output_shape, map_stride)`
///
/// - `output_shape` is the shape of the output ndarray after the reduction operation
///
/// - `map_stride` maps a flat iteration over the input ndarray to iteration over the output ndarray.
///   For example, if the reduce operation is addition, `reduce` iterates through the input ndarray
///   element-by-element and `map_stride` describes iteration over the output ndarray
///   to add each element to the correct location.
///   It should now make sense why map_stride contains zeros on every reduced axis
fn reduced_shape_and_stride(axes: &[isize], shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let ndims = shape.len();
    let mut axis_mask = vec![false; ndims];

    for &axis in axes.iter() {
        let axis = axis.as_absolute(ndims);
        if axis_mask[axis] {
            panic!("duplicate axes specified");
        }
        axis_mask[axis] = true;
    }

    let mut new_stride = VecDeque::with_capacity(ndims);
    let mut new_shape = VecDeque::with_capacity(ndims - axes.len());

    let mut stride = 1;
    for axis in (0..ndims).rev() {
        if axis_mask[axis] {
            new_stride.push_front(0);
        } else {
            new_stride.push_front(stride);
            new_shape.push_front(shape[axis]);
            stride *= shape[axis];
        }
    }

    (Vec::from(new_shape), Vec::from(new_stride))
}

impl<T: RawDataType> NdArray<'_, T> {
    /// Reduces the elements of a contiguous ndarray into a scalar using the specified function.
    ///
    /// # Safety
    /// - Ensure that the underlying ndarray has uniform stride in memory
    ///
    /// # Parameters
    /// - `func`: A closure or function that takes two arguments (the next value to be reduced
    ///   and the value of the accumulator) and returns a reduction of both.
    ///   For example, when the reduction operation is addition, `|src, acc| src + acc`
    /// - `default`: The initial value used as the accumulator for the reduction.
    /// - `stride`: The number of `T` elements in memory between consecutive elements of `self`
    unsafe fn reduce_uniform_stride(&self, func: impl Fn(T, T) -> T, default: T, stride: usize) -> NdArray<'static, T> {
        let mut output = default;

        let mut src = self.ptr();
        for _ in 0..self.size() {
            output = func(*src, output);
            src = src.add(stride);
        }

        NdArray::scalar(output)
    }

    fn reduce_along(&self, func: impl Fn(T, T) -> T, axes: impl ToVec<isize>, default: T) -> NdArray<'static, T> {
        let (out_shape, map_stride) = reduced_shape_and_stride(&axes.to_vec(), &self.shape);
        let (map_shape, map_stride) = collapse_to_uniform_stride(&self.shape, &map_stride);

        let mut output = vec![default; out_shape.iter().product()];

        let mut dst_indices = FlatIndexGenerator::from(&map_shape, &map_stride);
        let dst: *mut T = output.as_mut_ptr();

        for el in self.flatiter() {
            unsafe {
                let dst_i = dst_indices.next().unwrap();
                let dst_ptr = dst.add(dst_i);
                *dst_ptr = func(el, *dst_ptr);
            }
        }

        unsafe { NdArray::from_contiguous_owned_buffer(out_shape, output) }
    }

    fn reduce(&self, func: impl Fn(T, T) -> T, default: T) -> NdArray<'static, T> {
        if let Some(stride) = self.has_uniform_stride() {
            return unsafe { self.reduce_uniform_stride(func, default, stride) };
        }

        let mut output = default;

        for el in self.flatiter() {
            output = func(el, output);
        }

        NdArray::scalar(output)
    }
}

impl<T: NumericDataType> NdArray<'_, T> {
    /// Computes the sum of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![1, 2, 3, 4]);
    /// let sum = array.sum();
    /// assert_eq!(sum.value(), 1 + 2 + 3 + 4);
    /// ```
    pub fn sum(&self) -> NdArray<'static, T> {
        let output = unsafe { <T as ReduceSum>::sum(self.ptr(), self.shape(), self.stride()) };
        NdArray::scalar(output)
    }

    pub fn sum_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T> {
        self.reduce_along(|val, acc| acc + val, axes, T::zero())
    }

    /// Computes the product of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![1, 2, 3, 4]);
    /// let prod = array.product();
    /// assert_eq!(prod.value(), 1 * 2 * 3 * 4);
    /// ```
    pub fn product(&self) -> NdArray<'static, T> {
        let output = unsafe { <T as ReduceProduct>::product(self.ptr(), self.shape(), self.stride()) };
        NdArray::scalar(output)
    }

    pub fn product_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T> {
        self.reduce_along(|val, acc| acc * val, axes, T::one())
    }

    /// Computes the minimum of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![-1, 3, -7, 8]);
    /// let min = array.min();
    /// assert_eq!(min.value(), -7);
    /// ```
    pub fn min(&self) -> NdArray<'static, T> {
        let output = unsafe { <T as ReduceMin>::min(self.ptr(), self.shape(), self.stride()) };
        NdArray::scalar(output)
    }

    pub fn min_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T> {
        self.reduce_along(partial_min, axes, T::max_value())
    }

    /// Computes the maximum of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![-1, 3, -7, 8]);
    /// let max = array.max();
    /// assert_eq!(max.value(), 8);
    /// ```
    pub fn max(&self) -> NdArray<'static, T> {
        let output = unsafe { <T as ReduceMax>::max(self.ptr(), self.shape(), self.stride()) };
        NdArray::scalar(output)
    }

    pub fn max_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T> {
        self.reduce_along(partial_max, axes, T::min_value())
    }

    /// Computes the minimum absolute value of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![-1, 3, -7, 8]);
    /// let min = array.min_magnitude();
    /// assert_eq!(min.value(), 1);
    /// ```
    pub fn min_magnitude(&self) -> NdArray<'static, T> {
        let output = unsafe { <T as ReduceMinMagnitude>::min_magnitude(self.ptr(), self.shape(), self.stride()) };
        NdArray::scalar(output)
    }

    pub fn min_magnitude_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T> {
        self.reduce_along(partial_min_magnitude, axes, T::max_value())
    }

    /// Computes the maximum absolute value of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![-1, 3, -9, 8]);
    /// let max = array.max_magnitude();
    /// assert_eq!(max.value(), 9);
    /// ```
    pub fn max_magnitude(&self) -> NdArray<'static, T> {
        let output = unsafe { <T as ReduceMaxMagnitude>::max_magnitude(self.ptr(), self.shape(), self.stride()) };
        NdArray::scalar(output)
    }

    pub fn max_magnitude_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T> {
        self.reduce_along(partial_max_magnitude, axes, T::zero())
    }

    /// Computes the mean of all elements in the array.
    ///
    /// # Example
    /// ```
    /// use redstone::*;
    ///
    /// let array = NdArray::new(vec![1.0, 3.0, 5.0, 7.0]);
    /// let mean = array.mean();
    /// assert_eq!(mean.value(), 4.0);
    /// ```
    pub fn mean(&self) -> NdArray<'static, T>
    where
        T: FloatDataType
    {
        let n: T = NumCast::from(self.size()).unwrap();
        self.sum() / n
    }

    pub fn mean_along(&self, axes: impl ToVec<isize>) -> NdArray<'static, T>
    where
        T: FloatDataType
    {
        let axes = axes.to_vec();

        let mut n = 1;
        for &axis in axes.iter() {
            assert!(axis >= 0, "negative axes are not currently supported");
            n *= self.shape()[axis as usize];
        }

        let n: T = NumCast::from(n).unwrap();
        self.sum_along(axes) / n
    }
}


#[cfg(test)]
mod tests {
    use super::reduced_shape_and_stride;

    #[test]
    fn test_reduce_shape_and_stride() {
        let shape = vec![3, 2];

        let correct_shape = vec![3];
        let correct_stride = vec![1, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![1], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let shape = vec![4, 2, 3];

        let correct_shape = vec![2, 3];
        let correct_stride = vec![0, 3, 1];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![0], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![4, 3];
        let correct_stride = vec![3, 0, 1];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![1], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![4, 2];
        let correct_stride = vec![2, 1, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![2], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![3];
        let correct_stride = vec![0, 0, 1];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![0, 1], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![2];
        let correct_stride = vec![0, 1, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![0, 2], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![4];
        let correct_stride = vec![1, 0, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![1, 2], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);
    }
}
