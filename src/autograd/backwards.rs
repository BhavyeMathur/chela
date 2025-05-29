use crate::gradient_function::{GradientFuncTrait, GradientFunction};
use crate::{FloatDataType, NdArray, TensorMethods};
use std::cell::RefCell;
use std::rc::Rc;
use crate::broadcast::get_broadcasted_axes;
use crate::reshape::Reshape;
use crate::util::to_vec::ToVec;

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

    lhs: NdArray<'a, T>,
    rhs: NdArray<'a, T>
}

pub(crate) struct DivBackwards<'a, T: FloatDataType> {
    next_functions: [GradientFunction<T>; 2],

    lhs_grad: NdArray<'a, T>,
    rhs_grad: NdArray<'a, T>,

    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>
}

pub(crate) struct NegBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
}

pub(crate) struct ReshapeBackwards<T: FloatDataType> {
    next_function: GradientFunction<T>,

    shape: Vec<usize>,
}

impl<T: FloatDataType> GradientFuncTrait<T> for AddBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = reduce_broadcasted_gradient(grad, &self.lhs_shape);
        let rhs_grad = reduce_broadcasted_gradient(grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
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
        let lhs_grad = &self.rhs * grad;
        let rhs_grad = &self.lhs * grad;

        let lhs_grad = reduce_broadcasted_gradient(&lhs_grad, &self.lhs.shape());
        let rhs_grad = reduce_broadcasted_gradient(&rhs_grad, &self.rhs.shape());

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for DivBackwards<'_, T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let lhs_grad = &self.lhs_grad * grad;
        let rhs_grad = &self.rhs_grad * grad;

        let lhs_grad = reduce_broadcasted_gradient(&lhs_grad, &self.lhs_shape);
        let rhs_grad = reduce_broadcasted_gradient(&rhs_grad, &self.rhs_shape);

        self.next_functions[0].borrow_mut().backward(&lhs_grad);
        self.next_functions[1].borrow_mut().backward(&rhs_grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for NegBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = -grad;
        let grad = reduce_broadcasted_gradient(&grad, &self.shape);

        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> GradientFuncTrait<T> for ReshapeBackwards<T> {
    fn backward(&mut self, grad: &NdArray<T>) {
        let grad = grad.reshape(&self.shape);
        self.next_function.borrow_mut().backward(&grad);
    }
}

impl<T: FloatDataType> AddBackwards<T> {
    pub(crate) fn new(lhs: &NdArray<T>, rhs: &NdArray<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec()
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> SubBackwards<T> {
    pub(crate) fn new(lhs: &NdArray<T>, rhs: &NdArray<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_functions: [lhs.get_grad_fn(), rhs.get_grad_fn()],

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec()
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> MulBackwards<'static, T> {
    pub(crate) fn new(lhs: &NdArray<T>, rhs: &NdArray<T>) -> GradientFunction<T> {
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
    pub(crate) fn new(lhs: &NdArray<T>, rhs: &NdArray<T>) -> GradientFunction<T> {
        let next_functions = [lhs.get_grad_fn(), rhs.get_grad_fn()];

        let mut lhs = lhs.view();
        let mut rhs = rhs.view();
        lhs.set_requires_grad(false);
        rhs.set_requires_grad(false);

        let one = NdArray::scalar_requires_grad(T::one(), false);

        let grad_fn = Self {
            next_functions,

            lhs_grad: &one / &rhs,
            rhs_grad: -&lhs / (&rhs * &rhs),

            lhs_shape: lhs.shape().to_vec(),
            rhs_shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> NegBackwards<T> {
    pub(crate) fn new(rhs: &NdArray<T>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: rhs.get_grad_fn(),
            shape: rhs.shape().to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}

impl<T: FloatDataType> ReshapeBackwards<T> {
    pub(crate) fn new(tensor: &NdArray<T>, new_shape: impl ToVec<usize>) -> GradientFunction<T> {
        let grad_fn = Self {
            next_function: tensor.get_grad_fn(),
            shape: new_shape.to_vec(),
        };

        Rc::new(RefCell::new(grad_fn))
    }
}
