use crate::gradient_function::{AccumulateGrad, GradientFunction};
use crate::ndarray::flags::NdArrayFlags;
use crate::{FloatDataType, RawDataType, NdArray, TensorMethods};
use crate::into_gradient::IntoTensor;

impl<T: RawDataType> NdArray<'_, T> {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        if self.requires_grad() {
            self.flags().contains(NdArrayFlags::UserCreated)
        } else {
            true
        }
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.flags().contains(NdArrayFlags::RequiresGrad)
    }
}

impl<'a, T: FloatDataType> NdArray<'a, T> {
    pub fn set_requires_grad(&mut self, requires_grad: bool) -> &mut Self {
        let required_grad = self.requires_grad();

        if requires_grad {
            self.flags |= NdArrayFlags::RequiresGrad;
        } else {
            self.flags -= NdArrayFlags::RequiresGrad;
        }

        if !required_grad && requires_grad {
            self.grad_fn = AccumulateGrad::new(self.shape().to_vec());
        }

        self
    }
    
    /// Retrieves the gradient function associated with the current object.
    ///
    /// This is `NoneBackwards` if the tensor has `requires_grad = false` 
    /// or `AccumulateBackwards` if the tensor is a leaf node.
    pub(crate) fn get_grad_fn(&'a self) -> GradientFunction<T> {
        self.grad_fn.clone()
    }

    /// Returns the gradient of the differentiated tensor with respect to `self`.
    ///
    /// This method returns a view into the gradient.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let mut a = NdArray::scalar(2.0f32);
    /// let b = NdArray::scalar(3.0);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), b);
    /// ```
    pub fn gradient(&'a self) -> Option<NdArray<'a, T>> {
        unsafe { (*self.grad_fn.as_ptr()).gradient() }
    }

    /// Sets the gradient of this tensor to zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let mut a = NdArray::scalar(2.0f32);
    /// let b = NdArray::scalar(3.0);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// a.zero_gradient();
    /// assert_eq!(a.gradient().unwrap(), NdArray::scalar(0.0));
    /// ```
    pub fn zero_gradient(&self) {
        if let Some(mut grad ) = self.gradient() {
            grad.zero();
        }
    }

    /// Computes the gradient of the `self` with respect to its leaf tensors.
    ///
    /// # Parameters
    ///
    /// - `gradient`: the gradient of the tensor being differentiated with respect to `self`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let mut a = NdArray::full(2f32, [3]);  // [2, 2, 2]
    /// let b = NdArray::from([3.0, 1.0, -1.0]);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward_with(NdArray::from([2.0, 1.0, 1.0]));
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), NdArray::from([6f32, 1.0, -1.0]));
    /// ```
    pub fn backward_with(&self, gradient: impl IntoTensor<'a, T>) {
        let gradient = gradient.as_tensor();
        assert_eq!(gradient.shape(), self.shape());

        self.grad_fn.borrow_mut().backward(&gradient);
    }

    /// Computes the gradient of the `self` with respect to its leaf tensors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let mut a = NdArray::full(2f32, [3]);  // [2, 2, 2]
    /// let b = NdArray::from([3.0, 1.0, -1.0]);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), NdArray::from([3f32, 1.0, -1.0]));
    /// ```
    pub fn backward(&self) {
        self.backward_with(NdArray::ones(self.shape()))
    }
}
