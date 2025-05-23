use crate::dtype::RawDataType;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::tensor::flags::TensorFlags;
use crate::Tensor;

pub trait TensorMethods {
    /// Returns the dimensions of the tensor along each axis.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([3, 4, 5]);
    /// assert_eq!(a.shape(), &[3]);
    ///
    /// let b = Tensor::from([[3], [5]]);
    /// assert_eq!(b.shape(), &[2, 1]);
    ///
    /// let c = Tensor::scalar(0);
    /// assert_eq!(c.shape(), &[]);
    /// ```
    fn shape(&self) -> &[usize];

    /// Returns the stride of the tensor.
    ///
    /// The stride represents the distance in memory between elements in a tensor along each axis.
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let a = Tensor::from([[3, 4], [5, 6]]);
    /// assert_eq!(a.stride(), &[2, 1]);
    /// ```
    fn stride(&self) -> &[usize];

    /// Returns the number of dimensions in the tensor.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([3, 4, 5]);
    /// assert_eq!(a.ndims(), 1);
    ///
    /// let b = Tensor::from([[3], [5]]);
    /// assert_eq!(b.ndims(), 2);
    ///
    /// let c = Tensor::scalar(0);
    /// assert_eq!(c.ndims(), 0);
    /// ```
    fn ndims(&self) -> usize {
        self.shape().len()
    }

    /// Returns the length along the first dimension of the tensor.
    /// If the tensor is a scalar, this returns 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chela::*;
    /// let a = Tensor::from([3, 4, 5]);
    /// assert_eq!(a.len(), 3);
    ///
    /// let b = Tensor::from([[3], [5]]);
    /// assert_eq!(b.len(), 2);
    ///
    /// let c = Tensor::scalar(0);
    /// assert_eq!(c.len(), 0);
    /// ```
    #[inline]
    fn len(&self) -> usize {
        if self.shape().is_empty() {
            return 0;
        }

        self.shape()[0]
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([3, 4, 5]);
    /// assert_eq!(a.size(), 3);
    ///
    /// let b = Tensor::from([[3], [5]]);
    /// assert_eq!(b.size(), 2);
    ///
    /// let c = Tensor::scalar(0);
    /// assert_eq!(c.size(), 1);
    /// ```
    #[inline]
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns flags containing information about various tensor metadata.
    fn flags(&self) -> TensorFlags;

    /// Returns whether this tensor is stored contiguously in memory.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([[3, 4], [5, 6]]);
    /// assert!(a.is_contiguous());
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert!(!b.is_contiguous());
    /// ```
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.flags().contains(TensorFlags::Contiguous)
    }

    /// Returns whether this tensor is slice of another tensor.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([[3, 4], [5, 6]]);
    /// assert!(!a.is_view());
    ///
    /// let b = a.slice_along(Axis(1), 0);
    /// assert!(b.is_view());
    /// ```
    #[inline]
    fn is_view(&self) -> bool {
        !self.flags().contains(TensorFlags::Owned)
    }

    /// If the elements of this tensor are stored in memory with a uniform distance between them,
    /// returns this distance.
    ///
    /// Contiguous tensors always have a uniform stride of 1.
    /// Tensor views may sometimes be uniformly strided.
    ///
    /// ```rust
    /// # use chela::*;
    /// let a = Tensor::from([[3, 4, 5], [6, 7, 8]]);
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
        if !self.flags().contains(TensorFlags::UniformStride) {
            return None;
        }

        if self.ndims() == 0 {
            return Some(0);
        }

        let (_, new_stride) = collapse_to_uniform_stride(self.shape(), self.stride());
        Some(new_stride[0])
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        if self.requires_grad() {
            self.flags().contains(TensorFlags::UserCreated)
        }
        else {
            true
        }
    }

    #[inline]
    fn requires_grad(&self) -> bool {
        self.flags().contains(TensorFlags::RequiresGrad)
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self;
}

impl<T: RawDataType> TensorMethods for Tensor<'_, T> {
    #[inline]
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    fn stride(&self) -> &[usize] {
        &self.stride
    }

    #[inline]
    fn flags(&self) -> TensorFlags {
        self.flags
    }

    #[inline]
    fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        if requires_grad {
            self.flags |= TensorFlags::RequiresGrad;
        }
        else {
            self.flags -= TensorFlags::RequiresGrad;
        }

        self
    }
}

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub(crate) unsafe fn mut_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) unsafe fn ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}
