use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::util::to_vec::ToVec;
use crate::{FloatDataType, NdArray, Reshape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct ReshapeBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
}

impl<T: FloatDataType> GradientFuncTrait<T> for ReshapeBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad.reshape(&self.shape);
        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> ReshapeBackwards<T> {
    pub(crate) fn new(tensor: &Tensor<T>, new_shape: impl ToVec<usize>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: tensor.get_grad_fn(),
            shape: new_shape.to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
