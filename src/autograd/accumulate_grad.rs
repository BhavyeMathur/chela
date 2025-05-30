use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{NdArray, Reshape, TensorDataType};
use std::cell::RefCell;
use std::rc::Rc;

/// The default backwards node for leaf NdArrays.
///
/// Accumulates the gradient of the function being differentiated with respect to `self`
/// into the `tensor_grad` attribute of this struct.
pub(crate) struct AccumulateGrad<T: TensorDataType> {
    gradient: NdArray<'static, T>,
}

impl<T: TensorDataType> GradientFuncTrait<T> for AccumulateGrad<T> {
    /// Accumulates the gradient of the ndarray being differentiated with respect to a leaf ndarray
    /// into `tensor_grad`
    ///
    /// # Parameters
    ///
    /// - `grad`: the gradient of the function being differentiated with respect to `self`.
    fn backward(&mut self, grad: &NdArray<T>) {
        self.gradient += grad;
    }

    fn gradient<'a>(&'a self) -> Option<NdArray<'a, T>> {
        Some((&self.gradient).view())
    }
}

impl<T: TensorDataType> AccumulateGrad<T> {
    pub(crate) fn new(shape: Vec<usize>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            gradient: NdArray::zeros(shape),
        }))
    }
}
