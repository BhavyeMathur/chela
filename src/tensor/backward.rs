use crate::gradient_function::GradientFunction;
use crate::into_gradient::IntoTensor;
use crate::{FloatDataType, Tensor, TensorMethods};

impl<'a, T: FloatDataType> Tensor<'a, T> {
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
    /// let mut a = Tensor::scalar(2.0f32);
    /// let b = Tensor::scalar(3.0);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), b);
    /// ```
    pub fn gradient(&'a self) -> Option<Tensor<'a, T>> {
        unsafe { (*self.grad_fn.as_ptr()).gradient() }
    }

    /// Sets the gradient of this tensor to zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chela::*;
    ///
    /// let mut a = Tensor::scalar(2.0f32);
    /// let b = Tensor::scalar(3.0);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// a.zero_gradient();
    /// assert_eq!(a.gradient().unwrap(), Tensor::scalar(0.0));
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
    /// let mut a = Tensor::full(2f32, [3]);  // [2, 2, 2]
    /// let b = Tensor::from([3.0, 1.0, -1.0]);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward_with(Tensor::from([2.0, 1.0, 1.0]));
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), Tensor::from([6f32, 1.0, -1.0]));
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
    /// let mut a = Tensor::full(2f32, [3]);  // [2, 2, 2]
    /// let b = Tensor::from([3.0, 1.0, -1.0]);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), Tensor::from([3f32, 1.0, -1.0]));
    /// ```
    pub fn backward(&self) {
        self.backward_with(Tensor::ones(self.shape()))
    }
}
