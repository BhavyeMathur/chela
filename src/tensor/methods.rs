use std::rc::Rc;
use crate::ndarray::flags::NdArrayFlags;
use crate::common::methods::StridedMemory;
use crate::{NdArray, Tensor, TensorDataType};

impl<'a, T: TensorDataType> Tensor<'a, T> {
    /// Retrieves the single value contained within a tensor with a singular element.
    ///
    /// # Panics
    /// If the tensor contains more than one element (i.e., it is not a scalar or a tensor with a
    /// single element)
    ///
    /// # Example
    /// ```ignore
    /// # use chela::*;
    ///
    /// let tensor = Tensor::scalar(50.0);
    /// let value = tensor.value();
    /// assert_eq!(value, 50.0);
    /// ```
    ///
    /// # Notes
    /// This function is only meant for arrays that are guaranteed to have
    /// exactly one element. For arrays with multiple elements, consider using
    /// appropriate methods to access individual elements or slices safely.
    pub fn value(&self) -> T {
        self.array.value()
    }

    /// Returns a reference to the underlying `NdArray` of the tensor
    pub fn ndarray(&self) -> &NdArray<'a, T> {
        self.array.as_ref()
    }

    /// Converts the tensor to an `NdArray`
    pub fn into_ndarray(self) -> NdArray<'static, T> {
        match Rc::try_unwrap(self.array) {
            Ok(result) => { result }
            Err(rc) => { (*rc).clone() }
        }
    }
}

#[allow(clippy::len_without_is_empty)]
impl<'a, T: TensorDataType> StridedMemory for Tensor<'a, T> {
    /// Returns the dimensions of the tensor along each axis.
    ///
    /// ```ignore
    /// # use chela::*;
    ///
    /// let a = Tensor::from([3.0, 4.0, 5.0]);
    /// assert_eq!(a.shape(), &[3]);
    ///
    /// let b = Tensor::from([[3.0], [5.0]]);
    /// assert_eq!(b.shape(), &[2, 1]);
    ///
    /// let c = Tensor::scalar(0.0);
    /// assert_eq!(c.shape(), &[]);
    /// ```
    #[inline]
    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Returns the stride of the tensor.
    ///
    /// The stride represents the distance in memory between elements in a tensor along each axis.
    ///
    /// ```ignore
    /// # use chela::*;
    ///
    /// let a = Tensor::from([[3.0, 4.0], [5.0, 6.0]]);
    /// assert_eq!(a.stride(), &[2, 1]);
    /// ```
    #[inline]
    fn stride(&self) -> &[usize] {
        self.array.stride()
    }

    /// Returns flags containing information about various tensor metadata.
    #[inline]
    fn flags(&self) -> NdArrayFlags {
        self.flags
    }
}

#[allow(clippy::len_without_is_empty)]
impl<T: TensorDataType> StridedMemory for &Tensor<'_, T> {
    /// Returns the dimensions of the tensor along each axis.
    ///
    /// ```ignore
    /// # use chela::*;
    ///
    /// let a = Tensor::from([3.0, 4.0, 5.0]);
    /// assert_eq!(a.shape(), &[3]);
    ///
    /// let b = Tensor::from([[3.0], [5.0]]);
    /// assert_eq!(b.shape(), &[2, 1]);
    ///
    /// let c = Tensor::scalar(0.0);
    /// assert_eq!(c.shape(), &[]);
    /// ```
    #[inline]
    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Returns the stride of the tensor.
    ///
    /// The stride represents the distance in memory between elements in a tensor along each axis.
    ///
    /// ```ignore
    /// # use chela::*;
    ///
    /// let a = Tensor::from([[3.0, 4.0], [5.0, 6.0]]);
    /// assert_eq!(a.stride(), &[2, 1]);
    /// ```
    #[inline]
    fn stride(&self) -> &[usize] {
        self.array.stride()
    }

    /// Returns flags containing information about various tensor metadata.
    #[inline]
    fn flags(&self) -> NdArrayFlags {
        self.flags
    }
}
