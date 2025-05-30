use crate::ndarray::flags::NdArrayFlags;
use crate::traits::methods::StridedMemory;
use crate::{NdArray, Tensor, TensorDataType};

impl<'a, T: TensorDataType> Tensor<'a, T> {
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

    pub fn ndarray(&self) -> &NdArray<'a, T> {
        &self.array
    }

    pub fn to_ndarray(self) -> NdArray<'a, T> {
        self.array
    }
}

#[allow(clippy::len_without_is_empty)]
impl<T: TensorDataType> StridedMemory for Tensor<'_, T> {
    /// Returns the dimensions of the tensor along each axis.
    ///
    /// ```rust
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
    /// ```rust
    /// # use chela::*;
    ///
    /// let a = Tensor::from([[3.0, 4.0], [5.0, 6.0]]);
    /// assert_eq!(a.stride(), &[2, 1]);
    /// ```
    #[inline]
    fn stride(&self) -> &[usize] {
        self.array.stride()
    }

    /// Returns the number of dimensions in the tensor.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([3.0, 4.0, 5.0]);
    /// assert_eq!(a.ndims(), 1);
    ///
    /// let b = Tensor::from([[3.0], [5.0]]);
    /// assert_eq!(b.ndims(), 2);
    ///
    /// let c = Tensor::scalar(0.0);
    /// assert_eq!(c.ndims(), 0);
    /// ```
    fn ndims(&self) -> usize {
        self.array.ndims()
    }

    /// Returns the length along the first dimension of the tensor.
    /// If the tensor is a scalar, this returns 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chela::*;
    /// let a = Tensor::from([3.0, 4.0, 5.0]);
    /// assert_eq!(a.len(), 3);
    ///
    /// let b = Tensor::from([[3.0], [5.0]]);
    /// assert_eq!(b.len(), 2);
    ///
    /// let c = Tensor::scalar(0.0);
    /// assert_eq!(c.len(), 0);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.array.len()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([3.0, 4.0, 5.0]);
    /// assert_eq!(a.size(), 3);
    ///
    /// let b = Tensor::from([[3.0], [5.0]]);
    /// assert_eq!(b.size(), 2);
    ///
    /// let c = Tensor::scalar(0.0);
    /// assert_eq!(c.size(), 1);
    /// ```
    #[inline]
    fn size(&self) -> usize {
        self.array.size()
    }

    /// Returns flags containing information about various tensor metadata.
    #[inline]
    fn flags(&self) -> NdArrayFlags {
        self.flags
    }

    /// Returns whether this tensor is stored contiguously in memory.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([[3.0, 4.0], [5.0, 6.0]]);
    /// assert!(a.is_contiguous());
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert!(!b.is_contiguous());
    /// ```
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.array.is_contiguous()
    }

    /// Returns whether this tensor is slice of another tensor.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([[3.0, 4.0], [5.0, 6.0]]);
    /// assert!(!a.is_view());
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert!(b.is_view());
    /// ```
    #[inline]
    fn is_view(&self) -> bool {
        self.array.is_view()
    }

    /// If the elements of this tensor are stored in memory with a uniform distance between them,
    /// returns this distance.
    ///
    /// Contiguous tensors always have a uniform stride of 1.
    /// Tensor views may sometimes be uniformly strided.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
    /// assert_eq!(a.has_uniform_stride(), Some(1));
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert_eq!(b.has_uniform_stride(), Some(3));
    ///
    /// let c = a.slice_along(Axis(1), ..2);
    /// assert_eq!(c.has_uniform_stride(), None);
    /// ```
    #[inline]
    fn has_uniform_stride(&self) -> Option<usize> {
        self.array.has_uniform_stride()
    }
}
