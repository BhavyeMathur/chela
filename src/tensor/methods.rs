use crate::ndarray::flags::NdArrayFlags;
use crate::{NdArrayMethods, Tensor, TensorDataType};

#[allow(clippy::len_without_is_empty)]
impl<T: TensorDataType> Tensor<'_, T> {
    /// Returns the dimensions of the ndarray along each axis.
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let a = NdArray::from([3, 4, 5]);
    /// assert_eq!(a.shape(), &[3]);
    ///
    /// let b = NdArray::from([[3], [5]]);
    /// assert_eq!(b.shape(), &[2, 1]);
    ///
    /// let c = NdArray::scalar(0);
    /// assert_eq!(c.shape(), &[]);
    /// ```
    #[inline]
    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    /// Returns the stride of the ndarray.
    ///
    /// The stride represents the distance in memory between elements in an ndarray along each axis.
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let a = NdArray::from([[3, 4], [5, 6]]);
    /// assert_eq!(a.stride(), &[2, 1]);
    /// ```
    #[inline]
    fn stride(&self) -> &[usize] {
        self.array.stride()
    }

    /// Returns the number of dimensions in the ndarray.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = NdArray::from([3, 4, 5]);
    /// assert_eq!(a.ndims(), 1);
    ///
    /// let b = NdArray::from([[3], [5]]);
    /// assert_eq!(b.ndims(), 2);
    ///
    /// let c = NdArray::scalar(0);
    /// assert_eq!(c.ndims(), 0);
    /// ```
    fn ndims(&self) -> usize {
        self.array.ndims()
    }

    /// Returns the length along the first dimension of the ndarray.
    /// If the ndarray is a scalar, this returns 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chela::*;
    /// let a = NdArray::from([3, 4, 5]);
    /// assert_eq!(a.len(), 3);
    ///
    /// let b = NdArray::from([[3], [5]]);
    /// assert_eq!(b.len(), 2);
    ///
    /// let c = NdArray::scalar(0);
    /// assert_eq!(c.len(), 0);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.array.len()
    }

    /// Returns the total number of elements in the ndarray.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = NdArray::from([3, 4, 5]);
    /// assert_eq!(a.size(), 3);
    ///
    /// let b = NdArray::from([[3], [5]]);
    /// assert_eq!(b.size(), 2);
    ///
    /// let c = NdArray::scalar(0);
    /// assert_eq!(c.size(), 1);
    /// ```
    #[inline]
    fn size(&self) -> usize {
        self.array.size()
    }

    /// Returns flags containing information about various ndarray metadata.
    #[inline]
    fn flags(&self) -> NdArrayFlags {
        self.flags
    }

    /// Returns whether this ndarray is stored contiguously in memory.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = NdArray::from([[3, 4], [5, 6]]);
    /// assert!(a.is_contiguous());
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert!(!b.is_contiguous());
    /// ```
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.array.is_contiguous()
    }

    /// Returns whether this ndarray is slice of another ndarray.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = NdArray::from([[3, 4], [5, 6]]);
    /// assert!(!a.is_view());
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert!(b.is_view());
    /// ```
    #[inline]
    fn is_view(&self) -> bool {
        self.array.is_view()
    }

    /// If the elements of this ndarray are stored in memory with a uniform distance between them,
    /// returns this distance.
    ///
    /// Contiguous tensors always have a uniform stride of 1.
    /// NdArray views may sometimes be uniformly strided.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = NdArray::from([[3, 4, 5], [6, 7, 8]]);
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
