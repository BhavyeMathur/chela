use std::cell::RefCell;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, Tensor};
use std::rc::Rc;

pub(crate) struct AddBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],
}

pub(crate) struct SubBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],
}

pub(crate) struct MulBackwards<'a, T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs: Tensor<'a, T>,
    rhs: Tensor<'a, T>
}

pub(crate) struct DivBackwards<'a, T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_grad: Tensor<'a, T>,
    rhs_grad: Tensor<'a, T>
}

pub(crate) struct NegBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>
}

impl<'a, T: FloatDataType> GradientFuncTrait<T> for AddBackwards<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        self.next_functions[0].borrow_mut().backward(grad);
        self.next_functions[1].borrow_mut().backward(grad);
    }
}

impl<'a, T: FloatDataType> GradientFuncTrait<T> for SubBackwards<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        self.next_functions[0].borrow_mut().backward(grad);
        self.next_functions[1].borrow_mut().backward(&-grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for MulBackwards<'_, T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        let lhs_grad = &self.rhs * grad;
        let rhs_grad = &self.lhs * grad;

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for DivBackwards<'_, T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        let lhs_grad = &self.lhs_grad * grad;
        let rhs_grad = &self.rhs_grad * grad;

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for NegBackwards<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        self.next_function.borrow_mut().backward(&-grad);
    }
}

impl<T: FloatDataType> AddBackwards<T> {
    pub(crate) fn new(lhs: Tensor<T>, rhs: Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> SubBackwards<T> {
    pub(crate) fn new(lhs: Tensor<T>, rhs: Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> MulBackwards<'static, T> {
    pub(crate) fn new(lhs: Tensor<T>, rhs: Tensor<T>) -> GradientFunction<T> {
        let next_functions = [lhs.get_grad_fn(), rhs.get_grad_fn()];

        let mut lhs = lhs.clone();
        let mut rhs = rhs.clone();
        lhs.set_requires_grad(false);
        rhs.set_requires_grad(false);

        let grad_fn = Self {
            next_functions,
            lhs,
            rhs,
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> DivBackwards<'static, T> {
    pub(crate) fn new(lhs: Tensor<T>, rhs: Tensor<T>) -> GradientFunction<T> {

        let next_functions = [lhs.get_grad_fn(), rhs.get_grad_fn()];

        let mut lhs = lhs.clone();
        let mut rhs = rhs.clone();
        lhs.set_requires_grad(false);
        rhs.set_requires_grad(false);

        let one = Tensor::scalar_requires_grad(T::one(), false);
        let lhs_grad = &one / rhs;
        let rhs_grad = -lhs * (&lhs_grad * &lhs_grad);

        let grad_fn = Self {
            next_functions,
            lhs_grad,
            rhs_grad,
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> NegBackwards<T> {
    pub(crate) fn new(rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: rhs.get_grad_fn(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
