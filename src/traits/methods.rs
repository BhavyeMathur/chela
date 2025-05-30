use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::ndarray::flags::NdArrayFlags;

#[allow(clippy::len_without_is_empty)]
pub trait StridedMemory: Sized {
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
    fn shape(&self) -> &[usize];

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
    fn stride(&self) -> &[usize];

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
        self.shape().len()
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
        if self.shape().is_empty() {
            return 0;
        }

        self.shape()[0]
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
        self.shape().iter().product()
    }

    /// Returns flags containing information about various ndarray metadata.
    fn flags(&self) -> NdArrayFlags;

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
        self.flags().contains(NdArrayFlags::Contiguous)
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
        !self.flags().contains(NdArrayFlags::Owned)
    }

    /// If the elements of this ndarray are stored in memory with a uniform distance between them,
    /// returns this distance.
    ///
    /// Contiguous arrays always have a uniform stride of 1.
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
        if !self.flags().contains(NdArrayFlags::UniformStride) {
            return None;
        }

        if self.ndims() == 0 {
            return Some(0);
        }

        let (_, new_stride) = collapse_to_uniform_stride(self.shape(), self.stride());
        Some(new_stride[0])
    }
}
