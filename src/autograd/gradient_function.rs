use crate::{NumericDataType, RawDataType, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) type GradientFunction<T> = Rc<RefCell<dyn GradientFuncTrait<T>>>;

pub(crate) trait GradientFuncTrait<T: RawDataType> {
    fn backward(&mut self, grad: &Tensor<T>);

    fn gradient(&self) -> Option<Tensor<'static, T>> {
        None
    }

    fn zero(&mut self) {}
}

pub(crate) struct NoneBackwards {}

pub(crate) struct AccumulateGrad<T: NumericDataType> {
    tensor_grad: Tensor<'static, T>,
}

impl<T: RawDataType> GradientFuncTrait<T> for NoneBackwards {
    fn backward(&mut self, _: &Tensor<T>) {}
}

impl<T: NumericDataType> GradientFuncTrait<T> for AccumulateGrad<T> {
    fn backward(&mut self, grad: &Tensor<T>) {
        self.tensor_grad += grad;
    }

    fn gradient(&self) -> Option<Tensor<'static, T>> {
        Some(self.tensor_grad.clone())  // TODO can we get away without cloning?
    }

    fn zero(&mut self) {
        self.tensor_grad.zero();
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
