use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{Constructors, NdArray, Reshape, StridedMemory, TensorDataType};
use std::cell::RefCell;
use std::hint::assert_unchecked;
use std::rc::Rc;


/// The default backwards node for leaf Tensors.
///
/// Accumulates the gradient of the function being differentiated with respect to `self`
/// into the `tensor_grad` attribute of this struct.
pub(crate) struct AccumulateGrad<T: TensorDataType> {
    gradient: NdArray<'static, T>,
}

impl<T: TensorDataType> GradientFuncTrait<T> for AccumulateGrad<T> {
    /// Accumulates the gradient of the tensor being differentiated with respect to a leaf tensor
    /// into `tensor_grad`
    ///
    /// # Parameters
    ///
    /// - `grad`: the gradient of the function being differentiated with respect to `self`.
    fn backward(&mut self, grad: &NdArray<T>) {
        // hint that we don't need broadcasting
        unsafe { assert_unchecked(self.gradient.shape() == grad.shape()) }
        
        self.gradient += grad;
    }

    fn gradient(&self) -> Option<NdArray<T>> {
        Some((&self.gradient).view())
    }

    fn zero_gradient(&mut self) {
        self.gradient.zero();
    }
}

impl<T: TensorDataType> AccumulateGrad<T> {
    pub(crate) fn new(shape: Vec<usize>) -> GradientFunction<T> {
        Rc::new(RefCell::new(Self {
            gradient: NdArray::zeros(shape),
        }))
    }
}
