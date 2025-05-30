use crate::broadcast::get_broadcasted_axes;
use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::traits::Reshape;
use crate::{FloatDataType, NdArray, StridedMemory, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

fn reduce_broadcasted_gradient<'a, T: FloatDataType>(grad: &'a NdArray<'a, T>,
                                                     original_shape: &[usize]) -> NdArray<'a, T> {
    if grad.shape() == original_shape {
        return grad.view()
    }

    let axes = get_broadcasted_axes(grad.shape(), original_shape);
    let grad = grad.sum_along(axes);

    grad.reshape(original_shape)
}
pub(crate) struct AddBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>
}

pub(crate) struct SubBackwards<T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>
}

pub(crate) struct MulBackwards<'a, T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_grad: NdArray<'a, T>,
    rhs_grad: NdArray<'a, T>,

    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>
}

pub(crate) struct DivBackwards {}

pub(crate) struct AddScalarBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>
}

pub(crate) struct MulScalarBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
    scalar: T,
}

pub(crate) struct NegBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
}

pub(crate) struct DivScalarBackwards {}

impl<T: FloatDataType> GradientFuncTrait<T> for AddBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = reduce_broadcasted_gradient(grad, &self.lhs_shape);
        let rhs_grad = reduce_broadcasted_gradient(grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for AddScalarBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = reduce_broadcasted_gradient(grad, &self.shape);
        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for SubBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = grad;
        let rhs_grad = -grad;

        let lhs_grad = reduce_broadcasted_gradient(lhs_grad, &self.lhs_shape);
        let rhs_grad = reduce_broadcasted_gradient(&rhs_grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for MulBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = &self.lhs_grad * grad;
        let rhs_grad = &self.rhs_grad * grad;

        let lhs_grad = reduce_broadcasted_gradient(&lhs_grad, &self.lhs_shape);
        let rhs_grad = reduce_broadcasted_gradient(&rhs_grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for MulScalarBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad * self.scalar;
        let grad = reduce_broadcasted_gradient(&grad, &self.shape);

        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for NegBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = -grad;
        let grad = reduce_broadcasted_gradient(&grad, &self.shape);

        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> AddBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec()
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> AddScalarBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, _: T) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: lhs.get_grad_fn(),
            shape: lhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> SubBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec()
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> MulBackwards<'static, T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let next_functions = [lhs.get_grad_fn(), rhs.get_grad_fn()];

        let grad_fn = Self {
            next_functions,
            lhs_grad: rhs.detach(),
            rhs_grad: lhs.detach(),

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> MulScalarBackwards<T> {
    pub(crate) fn new(lhs: &Tensor<T>, rhs: T) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: lhs.get_grad_fn(),
            shape: lhs.shape().to_vec(),
            scalar: rhs
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl DivBackwards {
    pub(crate) fn new<T: FloatDataType>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> GradientFunction<T> {
        let next_functions = [lhs.get_grad_fn(), rhs.get_grad_fn()];

        let lhs = lhs.detach();
        let rhs = rhs.detach();

        let grad_fn = MulBackwards {
            next_functions,

            lhs_grad: NdArray::scalar(T::one()) / &rhs,
            rhs_grad: -&lhs / (&rhs * &rhs),

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl DivScalarBackwards {
    pub(crate) fn new<T: FloatDataType>(lhs: &Tensor<T>, rhs: T) -> GradientFunction<T> {
        MulScalarBackwards::new(lhs, T::one() / rhs)
    }
}

impl<T: FloatDataType> NegBackwards<T> {
    pub(crate) fn new(rhs: &Tensor<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: rhs.get_grad_fn(),
            shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
