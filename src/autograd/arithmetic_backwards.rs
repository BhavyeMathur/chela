use std::cell::RefCell;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{NumericDataType, Tensor};
use std::rc::Rc;

pub(crate) struct AddBackwards<T: NumericDataType> {
    next_functions: [GradientFunction<T>; 2],
}

impl<'a, T: NumericDataType> GradientFuncTrait<T> for AddBackwards<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        self.next_functions[0].borrow_mut().backward(grad);
        self.next_functions[1].borrow_mut().backward(grad);
    }
}

pub(crate) struct MultiplyBackwards<'a, T: NumericDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs: Tensor<'a, T>,
    rhs: Tensor<'a, T>
}

impl<T: NumericDataType> MultiplyBackwards<'static, T> {
    pub(crate) fn new(lhs: Tensor<T>, rhs: Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],

            lhs: lhs.clone(),
            rhs: rhs.clone(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: NumericDataType> GradientFuncTrait<T> for MultiplyBackwards<'_, T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        let lhs_grad = &self.rhs * grad;
        let rhs_grad = &self.lhs * grad;

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}
