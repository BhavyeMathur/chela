use crate::accumulate_grad::AccumulateGrad;
use crate::gradient_function::GradientFunction;
use crate::ndarray::flags::NdArrayFlags;
use crate::{Constructors, NdArray, StridedMemory, Tensor, TensorDataType};
use crate::none_backwards::NoneBackwards;

impl<'a, T: TensorDataType> Tensor<'a, T> {
    /// Checks if the tensor is a leaf.
    ///
    /// A tensor is considered a leaf node if `requires_grad = true`
    /// and it was explicitly created by the user, or if `requires_grad = false`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let mut tensor = Tensor::new([1.0, 2.0, 3.0]);
    /// tensor.set_requires_grad(true);
    /// assert!(tensor.is_leaf());
    ///
    /// let tensor2 = -tensor;
    /// assert!(!tensor2.is_leaf());
    /// ```
    #[inline]
    pub fn is_leaf(&self) -> bool {
        if self.requires_grad() {
            self.flags.contains(NdArrayFlags::UserCreated)
        } else {
            true
        }
    }

    /// Returns whether gradients must be computed for this tensor.
    ///
    /// A tensor is marked with the `requires_grad` flag if it was explicitly specified by the user
    /// through the `set_requires_grad()` method or if the tensor was created using operations
    /// on other tensors which were marked `requires_grad`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let mut tensor = Tensor::new([1.0, 2.0, 3.0]);
    /// tensor.set_requires_grad(true);
    ///
    /// let tensor2 = -tensor;
    /// assert!(tensor2.requires_grad());
    /// ```
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.flags.contains(NdArrayFlags::RequiresGrad)
    }

    /// Sets whether gradients must be computed for this tensor.
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
        if required_grad && !requires_grad {
            self.grad_fn = NoneBackwards::new();
        }

        self
    }

    /// Retrieves the gradient function associated with the current object.
    ///
    /// This is `NoneBackwards` if the tensor has `requires_grad = false`
    /// or `AccumulateBackwards` if the tensor is a leaf node.
    pub(crate) fn grad_fn(&self) -> GradientFunction<T> {
        self.grad_fn.clone()
    }

    /// Returns the gradient of the differentiated tensor with respect to `self`.
    ///
    /// This method returns a view into the gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
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
    pub fn gradient(&'a self) -> Option<NdArray<'a, T>> {
        unsafe { (*self.grad_fn.as_ptr()).gradient() }
    }

    /// Sets the gradient of this tensor to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
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
        self.grad_fn.borrow_mut().zero_gradient();
    }

    /// Computes the gradient of the `self` with respect to its leaf tensors.
    ///
    /// # Parameters
    ///
    /// - `gradient`: the gradient of the tensor being differentiated with respect to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let mut a = Tensor::full(2.0, [3]);  // [2, 2, 2]
    /// let b = Tensor::new([3.0, 1.0, -1.0]);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward_with(NdArray::new([2.0, 1.0, 1.0]));
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), Tensor::new([6.0, 1.0, -1.0]));
    /// ```
    pub fn backward_with(&self, gradient: impl AsRef<NdArray<'a, T>>) {
        let gradient = gradient.as_ref();
        assert_eq!(gradient.shape(), self.shape());

        self.grad_fn.borrow_mut().backward(gradient);
    }

    /// Computes the gradient of the `self` with respect to its leaf tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let mut a = Tensor::full(2.0, [3]);  // [2, 2, 2]
    /// let b = Tensor::new([3.0, 1.0, -1.0]);
    ///
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * &b;
    /// c.backward();
    ///
    /// // dc/da = b
    /// assert_eq!(a.gradient().unwrap(), Tensor::new([3.0, 1.0, -1.0]));
    /// ```
    pub fn backward(&self) {
        self.backward_with(NdArray::ones(self.shape()))
    }

    /// Detaches the tensor from the computation graph and returns an `NdArray`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use redstone_ml::*;
    ///
    /// let mut a = Tensor::full(2.0, [3]);
    /// a.set_requires_grad(true);
    ///
    /// let c = &a * 5.0;
    ///
    /// let d = c.detach();
    /// assert_eq!(d, NdArray::new([10.0, 10.0, 10.0]));
    /// ```
    pub fn detach(&self) -> NdArray<'static, T> {
        self.array.as_ref().clone()
    }
}
