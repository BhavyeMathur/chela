use crate::{NumericDataType, RawDataType, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) type GradientFunction<T> = Rc<RefCell<dyn GradientFuncTrait<T>>>;

pub(crate) trait GradientFuncTrait<T: RawDataType> {
    /// Computes the gradient of this function with respect to its sources using the chain rule.
    ///
    /// # Parameters
    ///
    /// - `grad`: the gradient of the function being differentiated with respect to `self`.
    fn backward(&mut self, grad: &Tensor<T>);

    /// Returns the gradient of the function being differentiated with respect to `self`
    /// if this function is a leaf. Otherwise, returns `None`.
    fn gradient<'a>(&'a self) -> Option<Tensor<'a, T>> {
        None
    }
}

/// The default backwards node for non-leaf Tensors.
pub(crate) struct NoneBackwards {}

/// The default backwards node for leaf Tensors.
/// 
/// Accumulates the gradient of the function being differentiated with respect to `self`
/// into the `tensor_grad` attribute of this struct.
pub(crate) struct AccumulateGrad<T: NumericDataType> {
    tensor_grad: Tensor<'static, T>,
}

impl<T: RawDataType> GradientFuncTrait<T> for NoneBackwards {
    /// Backwards method for tensor with `requires_grad = false`, does nothing.
    fn backward(&mut self, _: &Tensor<T>) {}
}

impl<T: NumericDataType> GradientFuncTrait<T> for AccumulateGrad<T> {
    /// Accumulates the gradient of the tensor being differentiated with respect to a leaf tensor
    /// into `tensor_grad`
    ///
    /// # Parameters
    ///
    /// - `grad`: the gradient of the function being differentiated with respect to `self`.
    fn backward(&mut self, grad: &Tensor<T>) {
        self.tensor_grad += grad;
    }

    fn gradient<'a>(&'a self) -> Option<Tensor<'a, T>> {
        Some((&self.tensor_grad).view())
    }
}

impl<T: NumericDataType> AccumulateGrad<T> {
    pub(crate) fn new(shape: Vec<usize>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            tensor_grad: Tensor::zeros(shape),
        }))
    }
}

impl NoneBackwards {
    pub(crate) fn new<T: RawDataType>() -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {}))
    }
}
